//TODO: impémenter MSE. Voir si fichier distinct 
use smallvec::SmallVec;

use crate::nn::functions;
use crate::nn::functions::apply;
use crate::ops::hadamard_mul;
use crate::tensor;
use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId, Node};
use crate::ops::hadamard_mul_direct;
use crate::ops::linalg::matmul;
use crate::ops::sub;
use core::f32;
use std::f32::NEG_INFINITY;
use std::result;
use std::sync::Arc; 
use std::ops::Add;
use smallvec::smallvec;
use crate::ops::shapes::mean_all;

/*
mse: sum (xi - xtilde i)^2 = hadamard_mul(diff diff)
*/
pub fn mse(tr: &mut Trace, pred_id: NodeId, target_id: NodeId) -> NodeId{ // A check si bien ok ? ca me parait chelou de faire 2 node pour ca.. un mse creerait donc 3 matrices ! ?  une pour la diff, une pour la transpoée et le résultat du produit... Peut être plus efficace en implémentant from scratch..
   

    let diff_id = sub(tr, pred_id, target_id);

    // on utilise pas hadamard mul car c'est un carré la..
    let square_id = functions::apply(tr, diff_id, |x| x*x, |x| 2f32*x);
    
    mean_all(tr, square_id)
}


pub fn softmax(t: &Tensor) -> (Tensor, Tensor){
    //let mx = t.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let n = t.shape.len();
    assert!(t.shape[n-1] > 0);
    
    // on a besoin du unsqueeze view a la fin pour que le broadcast se fasse bien sur ça. mais ca c'est déjà fait dans le code précédent 
    let m = t.max_last();


    // ca garde les strides de t. donc ok 
    let scaled = t- &m;

    let exp = scaled.apply(f32::exp);

    let s = exp.sum_last();

    let lse = &m+ &s.apply(f32::ln);
    let softmax = &exp/ &s; 
    (lse, softmax)

}
pub fn softmax_crossentropy(tr: &mut Trace, logits_id: NodeId, target_id: NodeId) -> NodeId{
    let logits = tr.get_tensor(logits_id);
    let y = tr.get_tensor(target_id);
    let (lse, softmaxed) = softmax(logits);

    // multiplication element apr element => sum last => moyenne pondérée du label voulu predit
    let zy = (logits*y).sum_last();

    let value = &lse - &zy; 


    let soft_c = softmaxed.clone();
    let y_c = y.clone();
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{

        let diff = &soft_c - &y_c;
        smallvec![(logits_id, &diff*g_out)]
    };
    let smxcpy = tr.push(Node { value: value, parents_id: smallvec![logits_id], vjp: Some(Box::new(vjp)), is_param: false });
    
    mean_all(tr, smxcpy)
}