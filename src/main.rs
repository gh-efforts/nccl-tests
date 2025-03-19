use anyhow::Result;
use candle_core::{CudaDevice, Device, Tensor};
use candle_core::backend::BackendDevice;

fn exec() -> Result<()>{
    let core_0 = CudaDevice::new(0)?;
    let core_1 = CudaDevice::new(1)?;

    let core_0 = Device::Cuda(core_0);
    let core_1 = Device::Cuda(core_1);

    let t1x8 = Tensor::arange(0f32, 8f32, &core_0)?;
    let t1x8 = t1x8.to_device(&core_1)?;
    t1x8.device().synchronize()?;

    Ok(())
}

fn main() {
    exec().unwrap();
}
