use crate::tensor::Tensor; 

pub struct Sgd{
    pub lr: f32
}

impl Sgd{
    pub fn update(&self, params: &[Tensor], grads: &[Tensor]) -> Vec<Tensor>{
        params.iter().zip(grads.iter())
            .map(|(param, grad)|
                param + &grad.apply(|x| x*(-self.lr))
        ).collect()
    }
}