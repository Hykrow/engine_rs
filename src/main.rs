use crate::tensor::Tensor;

mod tensor;
mod graph; 

mod ops;

use ops::Op;

fn main() {
    println!("Hello, world!");

    let a = Tensor::from_vec(&((1..=8).map(|x| x as f32).collect::<Vec<f32>>()), &[2, 4]).unwrap();
    let new_shape = Tensor::broadcast_shape(&[ 2, 4], &[2, 4]);
    let b = a.broadcast_view(&new_shape.unwrap()).unwrap();
    let c = b.mat_transpose();
    print!("mat initiale : {}, mat transposée : {} \n \n", b, b.mat_transpose());
    println!("{}", c.get2(0, 1));

    let parent1 = Tensor::from_vec(&[1f32, 2f32, 3f32, 4f32], &[2, 2]).unwrap();

    let parent2 = Tensor::from_vec(&[4f32], &[1]).unwrap();

    print!("doing multiplication.");
    let child = Op::MatMul.forward(&[&parent1, &parent2]);


    let before_grad = Tensor::from_vec(&[1f32, 0f32, 0f32, 1f32], &[2, 2]).unwrap();
    print!("parent 1 : {} \n parent 2 : {} \n before grad :  {} , child : {} \n", parent1, parent2, before_grad, child);


    /*
    let grad_left = ops::tensor_mul(&before_grad, &parent2.mat_transpose());
    let grad_right = ops::tensor_mul(&parent1.mat_transpose(), &before_grad);
    print!("{}, \n {}", grad_left, grad_right);
    */
}
