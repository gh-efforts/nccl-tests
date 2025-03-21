use std::time::Instant;
use anyhow::{anyhow, Result};
use candle_core::{CudaDevice, Device, Tensor};
use candle_core::backend::BackendDevice;

// same node, same numa, GPU0 -> CPU mem -> GPU1
fn t1(n: f32) -> Result<()>{
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(1)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let x = Tensor::arange(0f32, n, &core_0)?;
    x.device().synchronize()?;

    let t = Instant::now();
    let x= x.to_device(&core_1)?;
    x.device().synchronize()?;

    println!("same node, GPU0 -> CPU mem -> GPU1, use {:?}", t.elapsed());
    Ok(())
}

// same node, cross numa, GPU0 -> CPU mem -> GPU1
fn t2(n: f32) -> Result<()>{
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(7)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let x = Tensor::arange(0f32, n, &core_0)?;
    x.device().synchronize()?;

    let t = Instant::now();
    let x= x.to_device(&core_1)?;
    x.device().synchronize()?;

    println!("same node, GPU0 -> CPU mem -> GPU7, use {:?}", t.elapsed());
    Ok(())
}

fn t3(n: f32) -> Result<()> {
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(1)?;

    let core_0 = core_0.cuda_device();
    let core_1 = core_1.cuda_device();

    let devices = vec![core_0, core_1];
    let comms = cudarc::nccl::Comm::from_devices(devices).map_err(|e| anyhow!("{:?}", e))?;

    let s = comms[0].world_size();
    println!("{}", s);
    Ok(())
}

fn main() {
    t3(10f32).unwrap();
}
