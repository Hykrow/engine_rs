use rand::Rng;
use crate::tensor::Tensor;

pub fn xavier_uniform(in_dim: usize, out_dim: usize)-> Tensor{
    let limit = f32::sqrt(6f32/(in_dim as f32 +out_dim as f32 ));
    let mut rng = rand::thread_rng();
    let mut vec = Vec::with_capacity(in_dim*out_dim);
    for _ in 0..in_dim*out_dim{
        vec.push(rng.gen_range((-limit)..(limit)));
    }
    Tensor::from_vec(&vec, &[in_dim, out_dim]).unwrap()
} 

pub fn kaiming(in_dim: usize, out_dim: usize) -> Tensor{
    let limit = f32::sqrt(6f32/(in_dim as f32));
    let mut rng = rand::thread_rng();
    let mut vec = Vec::with_capacity(in_dim*out_dim);
    for _ in 0..in_dim*out_dim{
        vec.push(rng.gen_range((-limit)..(limit)));
    }
    Tensor::from_vec(&vec, &[in_dim, out_dim]).unwrap()
}