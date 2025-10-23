use crate::tensor::Tensor;

mod tensor;
mod graph; 

mod ops;

fn main() {
    println!("Hello, world!");

    let a = Tensor::ones(&[2, 2, 3]);
    let new_shape = Tensor::broadcast_shape(&[5, 2, 4], &[3, 4]);
    println!("{:?}", new_shape.unwrap());
    }
