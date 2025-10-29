
use lamp::autodiff::value_and_grad::{value_and_grad};
use lamp::nn::losses::mse;
use lamp::nn::functions::tanh;
use lamp::ops::matmul; 
use lamp::optim::sgd;
use lamp::tensor::Tensor;
use lamp::trace::{Trace};

fn main() {
    let mut tr = Trace::new();

    let mut params = vec![
        Tensor::random(&[5, 2, 2], 1f32)
    ];
    let sgd = sgd::Sgd {lr: 0.1};




    for _ in 0..10{
    // vrm débile, aucun sens en soi..
        let (loss, grads) = value_and_grad(&params, |tr, pids|{

                let x = tr.input(Tensor::ones(&[2])); 
                let y = tr.input(Tensor::ones(&[2]));

                let multiplied = matmul(tr, x, pids[0]); // (5, 2, 2) @ (2) => (5, 2)
                let multiplied_tanh = tanh(tr, multiplied); // (5, 2)
                mse(tr, multiplied_tanh, y)
        });
        print!("loss: {}", loss.data[0]);
        params = sgd.update(&params, &grads);
    }



    
}
