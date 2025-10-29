//TODO: impémenter MSE. Voir si fichier distinct 
use smallvec::SmallVec;

use crate::ops::hadamard_mul;
use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId, Node};
use crate::ops::hadamard_mul_direct;
use crate::ops::linalg::matmul;
use crate::ops::sub;
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

    let prod_id = hadamard_mul(tr, diff_id, diff_id);
    
    mean_all(tr, prod_id)
}



pub fn softmax_crossentropy(tr: &mut Trace, pred_id: NodeId, target_id: NodeId) -> NodeId{
    let sum_exp = tr.get_tensor(pred_id).apply(f32::exp).data.iter().sum::<f32>();
    let softmax = tr.get_tensor(pred_id).apply(|x| f32::exp(x)/sum_exp);
    // faut il broadcast, faut il sommer puis divisier ou diviser puis sommer ? je pense équivalent

    
}