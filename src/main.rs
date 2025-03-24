use anyhow::{anyhow, ensure, Result};
use candle_core::backend::BackendDevice;
use candle_core::cuda::CudaStorageSlice;
use candle_core::{CpuStorage, CudaDevice, CudaStorage, DType, Device, InplaceOp1, Layout, Shape, Storage, Tensor};
use cudarc::nccl::result::NcclStatus;
use cudarc::nccl::{Comm, Id};
use std::sync::Arc;
use std::time::Instant;

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

    let core_0 = CudaDevice::new(0)?;
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

fn main() {
    t1((2, 4), 0, 1).unwrap();
    t1((2048, 4096), 0, 1).unwrap();
    t1((2, 4), 0, 7).unwrap();
    t1((2048, 4096), 0, 7).unwrap();


    t2((2, 4), 0, 1).unwrap();
    t2((2048, 4096), 0, 1).unwrap();
    t2((2, 4), 0, 7).unwrap();
    t2((2048, 4096), 0, 7).unwrap();
}
