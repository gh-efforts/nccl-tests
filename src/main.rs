use std::time::Instant;
use anyhow::Result;
use candle_core::{CudaDevice, Device, Tensor};
use candle_core::backend::BackendDevice;

// same node, GPU0 -> CPU mem -> GPU1
fn t1() -> Result<()>{
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(1)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let t1x8 = Tensor::arange(0f32, 64f32, &core_0)?;

    let t = Instant::now();
    let t1x8 = t1x8.to_device(&core_1)?;
    t1x8.device().synchronize()?;

    println!("use {:?}", t.elapsed());
    Ok(())
}

fn main() {
    t1().unwrap();
}
