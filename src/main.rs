use anyhow::{anyhow, ensure, Result};
use candle_core::backend::BackendDevice;
use candle_core::cuda::CudaStorageSlice;
use candle_core::{CpuStorage, CudaDevice, CudaStorage, DType, Device, InplaceOp1, Layout, Shape, Storage, Tensor};
use cudarc::nccl::result::NcclStatus;
use cudarc::nccl::{Comm, Id};
use std::env::args;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;
use zerocopy::IntoBytes;

struct TensorCopy<'a> {
    comm: &'a Comm,
    from: i32,
}

impl InplaceOp1 for TensorCopy<'_> {
    fn name(&self) -> &'static str {
        "tensor-copy"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> candle_core::Result<()> {
        unimplemented!();
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, _layout: &Layout) -> candle_core::Result<()> {
        let slice = match &mut storage.slice {
            CudaStorageSlice::F32(v) => v,
            _ => unreachable!()
        };

        while self.comm
            .recv(slice, self.from)
            .map_err(|_| candle_core::Error::msg("nccl error"))? == NcclStatus::InProgress {}

        Ok(())
    }
}

// same node, GPU0 -> CPU mem -> GPU1
fn t1<S: Into<Shape> + Copy>(shape: S, core0: usize, core1: usize) -> Result<()> {
    let core_0 = CudaDevice::new(core0)?;
    let core_1 = CudaDevice::new(core1)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let x_0 = Tensor::rand::<_, f32>(0f32, 100f32, shape.clone(), &core_0)?;
    x_0.device().synchronize()?;

    let a = Tensor::full(1f32, shape, &core_1)?;
    a.device().synchronize()?;

    let t = Instant::now();
    let x_1 = x_0.to_device(&core_1)?;
    let x_1 = x_1.add(&a)?;
    let new_x_0 = x_1.to_device(&core_0)?;
    new_x_0.device().synchronize()?;
    let elapsed = t.elapsed();
    println!("same node, shape: {:?}, dtype: {}, CORE{} -> CPU mem -> CORE{}, use {:?}", new_x_0.shape(), "f32", core0, core1, elapsed);

    let a_0 = a.to_device(&core_0)?;
    let x_0 = x_0.add(&a_0)?;

    let new_x_0 = new_x_0.to_string();
    let x_0 = x_0.to_string();
    ensure!(new_x_0 == x_0);
    Ok(())
}

fn t2<S: Into<Shape> + Copy + Send + 'static>(shape: S, core0: usize, core1: usize) -> Result<()> {
    let id = Id::new().unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(2));

    std::thread::spawn({
        let barrier = barrier.clone();

        move || {
            let core_1 = CudaDevice::new(core1)?;
            let core_1_raw = core_1.cuda_device();
            let comm = Comm::from_rank(core_1_raw, 1, 2, id).map_err(|_| anyhow!("nccl error"))?;
            let core_1 = Device::Cuda(core_1);

            let a = Tensor::full(1f32, shape, &core_1)?;
            a.device().synchronize()?;

            let mut op = TensorCopy { comm: &comm, from: 0 };
            let t = Tensor::zeros(shape, DType::F32, &core_1)?;
            t.device().synchronize()?;

            barrier.wait();

            t.inplace_op1(&mut op)?;
            let out = t.add(&a)?;
            out.device().synchronize()?;
            let (data, _) = out.storage_and_layout();
            let s = match &(*data) {
                Storage::Cuda(s) => s,
                _ => unreachable!(),
            };
            let data = s.as_cuda_slice::<f32>()?;
            comm.send(data, 0)
                .map_err(|e| anyhow!("{:?}", e))?;

            Result::<_, anyhow::Error>::Ok(())
        }
    });

    let core_0 = CudaDevice::new(core0)?;
    let core_0_raw = core_0.cuda_device();
    let core_0 = Device::Cuda(core_0);
    let comm = Comm::from_rank(core_0_raw, 0, 2, id).unwrap();

    let mut op = TensorCopy { comm: &comm, from: 1 };

    let x = Tensor::rand::<_, f32>(0f32, 100f32, shape.clone(), &core_0)?;
    x.device().synchronize()?;

    let recv_t = Tensor::zeros(shape, DType::F32, &core_0)?;
    recv_t.device().synchronize()?;

    barrier.wait();

    let t = Instant::now();
    let (data, _layout) = x.storage_and_layout();
    let s = match &(*data) {
        Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let data = s.as_cuda_slice::<f32>()?;
    comm
        .send(data, 1)
        .map_err(|e| anyhow!("{:?}", e))?;
    recv_t.inplace_op1(&mut op)?;
    recv_t.device().synchronize()?;
    let elapsed = t.elapsed();

    let a = Tensor::full(1f32, shape, &core_0)?;
    let x = x.add(&a)?;

    ensure!(recv_t.to_string() == x.to_string());
    println!("same node use nccl, shape {:?}, dtype {}, CORE{} -> CORE{}, use {:?}", x.shape(), "f32", core0, core1, elapsed);
    Ok(())
}

fn t3_master<S: Into<Shape> + Copy + Send + 'static>(shape: S, core0: usize, mirror_addr: &str) -> Result<()> {
    let id = Id::new().unwrap();
    let id_bytes = id.internal();
    let id_bytes = id_bytes.as_bytes();

    let mut stream = TcpStream::connect(mirror_addr).unwrap();
    stream.write_all(id_bytes)?;
    let shape = shape.into();
    let shape_buff = serde_json::to_vec(&shape.dims()).unwrap();
    let shape_len = shape_buff.len() as u32;
    stream.write_all(&shape_len.to_be_bytes())?;
    stream.write_all(shape_buff.as_slice())?;

    let core_0 = CudaDevice::new(core0)?;
    let core_0_raw = core_0.cuda_device();
    let core_0 = Device::Cuda(core_0);
    println!("before comm init");
    let comm = match Comm::from_rank(core_0_raw, 0, 2, id) {
        Ok(comm) => comm,
        Err(e) => {
            eprintln!("nccl err: {:?}", e.0);
            panic!("nccl err");
        }
    };

    println!("after comm init");
    let mut op = TensorCopy { comm: &comm, from: 1 };

    let x = Tensor::rand::<_, f32>(0f32, 100f32, shape.clone(), &core_0)?;
    x.device().synchronize()?;

    let recv_t = Tensor::zeros(shape.clone(), DType::F32, &core_0)?;
    recv_t.device().synchronize()?;

    let t = Instant::now();
    let (data, _layout) = x.storage_and_layout();
    let s = match &(*data) {
        Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let data = s.as_cuda_slice::<f32>()?;
    println!("before send");
    comm
        .send(data, 1)
        .map_err(|e| anyhow!("{:?}", e))?;
    println!("after send");
    recv_t.inplace_op1(&mut op)?;
    println!("after recv");
    // recv_t.device().synchronize()?;
    // println!("after sync");
    let elapsed = t.elapsed();

    let a = Tensor::full(1f32, shape, &core_0)?;
    let x = x.add(&a)?;

    ensure!(recv_t.to_string() == x.to_string());
    println!("use nccl, shape {:?}, dtype {}, node0 -> node1, use {:?}", x.shape(), "f32", elapsed);
    Ok(())
}

fn t3_mirror(core1: usize) -> Result<()> {
    let listener = TcpListener::bind("0.0.0.0:34053")?;

    loop {
        let (mut stream, _) = listener.accept()?;
        let mut id_buf = [0u8; 128];
        let mut id_arr = [0i8; 128];
        stream.read_exact(&mut id_buf)?;
        for (i, &data) in id_buf.iter().enumerate() {
            id_arr[i] = data as i8;
        }
        let id = Id::uninit(id_arr);
        println!("id: {:?}", id);

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut shape_buf = vec![0u8; len];
        stream.read_exact(&mut shape_buf)?;
        let shape: Vec<usize> = serde_json::from_slice(&shape_buf)?;

        let core_1 = CudaDevice::new(core1)?;
        let core_1_raw = core_1.cuda_device();

        let core_1 = Device::Cuda(core_1);

        let a = Tensor::full(1f32, shape.clone(), &core_1)?;
        a.device().synchronize()?;

        let t = Tensor::zeros(shape, DType::F32, &core_1)?;
        t.device().synchronize()?;

        println!("before comm init");

        let comm = match Comm::from_rank(core_1_raw, 1, 2, id) {
            Ok(comm) => comm,
            Err(e) => {
                eprintln!("nccl err: {:?}", e.0);
                panic!("nccl err");
            }
        };
        println!("after comm init");

        let mut op = TensorCopy { comm: &comm, from: 0 };

        t.inplace_op1(&mut op)?;
        let out = t.add(&a)?;
        // out.device().synchronize()?;
        let (data, _) = out.storage_and_layout();
        let s = match &(*data) {
            Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        let data = s.as_cuda_slice::<f32>()?;
        println!("before comm send");
        comm.send(data, 0)
            .map_err(|e| anyhow!("{:?}", e))?;
        println!("after commd send")
    }
}

fn main() {
    let mut args = args();
    args.next();

    match args.next().as_deref() {
        Some("master") => {
            // t1((2, 4), 0, 1).unwrap();
            // t1((2048, 4096), 0, 1).unwrap();
            // t1((2048 * 8, 4096 * 8), 0, 1).unwrap();
            //
            // t1((2, 4), 0, 7).unwrap();
            // t1((2048, 4096), 0, 7).unwrap();
            // t1((2048 * 8, 4096 * 8), 0, 7).unwrap();
            //
            // t2((2, 4), 0, 1).unwrap();
            // t2((2048, 4096), 0, 1).unwrap();
            // t2((2048 * 8, 4096 * 8), 0, 1).unwrap();
            //
            // t2((2, 4), 0, 7).unwrap();
            // t2((2048, 4096), 0, 7).unwrap();
            // t2((2048 * 8, 4096 * 8), 0, 7).unwrap();

            let mirror_addr = args.next().unwrap();
            t3_master((2, 4), 3, &mirror_addr).unwrap();
            t3_master((2048, 4096), 3, &mirror_addr).unwrap();
            t3_master((2048 * 8, 4096 * 8), 3, &mirror_addr).unwrap();
        }
        Some("mirror") => {
            std::thread::scope(|s| {
                t3_mirror(0).unwrap();
            })
        }
        _ => panic!("unknown command"),
    }
}
