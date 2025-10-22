use crate::tensor::Tensor;

mod tensor;
mod graph; 
mod ops; 
fn main() {
    println!("Hello, world!");

    let t = Tensor::ones(&[2, 2, 3]);
    println!("{:?}", t.strides);
}
