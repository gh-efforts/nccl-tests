use anyhow::{anyhow, Result};
use candle_core::backend::BackendDevice;
use candle_core::{CpuStorage, CudaDevice, CudaStorage, DType, Device, InplaceOp1, Layout, Shape, Storage, Tensor};
use cudarc::nccl::{Comm, Id};
use std::time::Instant;
use candle_core::cuda::CudaStorageSlice;
use cudarc::nccl::result::NcclError;

struct TensorCopy {
    comm: Comm,
    from: i32,
}

impl InplaceOp1 for TensorCopy {
    fn name(&self) -> &'static str {
        "tensor-copy"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        todo!()
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, _layout: &Layout) -> candle_core::Result<()> {
        let slice = match &mut storage.slice {
            CudaStorageSlice::F32(v) => v,
            _ => unreachable!()
        };

        self.comm
            .recv(slice, self.from)
            .unwrap();
        Ok(())
    }
}

// same node, same numa, GPU0 -> CPU mem -> GPU1
fn t1<S: Into<Shape> + Clone>(shape: S) -> Result<()> {
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(1)?;

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
    println!("same node, GPU0 -> CPU mem -> GPU1, use {:?}", elapsed);

    let a_0 = a.to_device(&core_0)?;
    let x_0 = x_0.add(&a)?;

    new_x_0.eq(&x_0)?;
    Ok(())
}

// same node, cross numa, GPU0 -> CPU mem -> GPU1
fn t2(n: f32) -> Result<()> {
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(7)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let x = Tensor::arange(0f32, n, &core_0)?;
    x.device().synchronize()?;

    let t = Instant::now();
    let x = x.to_device(&core_1)?;
    x.device().synchronize()?;

    println!("same node, GPU0 -> CPU mem -> GPU7, use {:?}", t.elapsed());
    Ok(())
}

fn t3(n: f32) -> Result<()> {
    let id = Id::new().unwrap();

    let core_0 = CudaDevice::new(0)?;
    let core_0_raw = core_0.cuda_device();
    let core_0 = Device::Cuda(core_0);

    let x = Tensor::arange(0f32, n, &core_0)?;
    let x_count = x.elem_count();

    let h = std::thread::spawn({
        move || {
            let core_1 = CudaDevice::new(1)?;
            let core_1_raw = core_1.cuda_device();

            let res = cudarc::nccl::Comm::from_rank(core_1_raw, 1, 2, id);
            let comm = match res {
                Ok(comm) => comm,
                Err(e) => {
                    eprintln!("{:?}", e.0);
                    panic!("{:?}", e);
                }
            };

            let core_1 = Device::Cuda(core_1);
            let mut op = TensorCopy { comm, from: 0 };
            let t = Tensor::zeros((x_count, 1), DType::F32, &core_1)?;
            println!("wait recv");
            t.inplace_op1(&mut op)?;
            println!("wait sync");
            t.device().synchronize()?;

            println!("{}", t);
            Result::<_, anyhow::Error>::Ok(())
        }
    });

    println!("before init rank 0");
    let comm = cudarc::nccl::Comm::from_rank(core_0_raw, 0, 2, id).unwrap();
    println!("after init rank 0");

    let (data, layout) = x.storage_and_layout();

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

    h.join().unwrap().unwrap();
    Ok(())
}

fn main() {
    t1((2, 4)).unwrap();
}
