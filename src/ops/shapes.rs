use smallvec::{smallvec, SmallVec};

use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId};

pub fn mean_all(tr: &mut Trace, x_id: NodeId) -> NodeId{
    let x = tr.get_tensor(x_id).clone(); 
    let n = x.shape.numel();
    let y = x.sum_all(); 


    /*
    d/dx mean x = 1/n. 
     */
 //   println!("X SHAPE :  {:?}", x.shape);
    let vjp = move |g_out: &Tensor|  -> SmallVec<[(NodeId, Tensor); 2]>{
        let gx = g_out.apply(|x| x/(n as f32)).broadcast_view(&x.shape).unwrap();
   //     println!("G X, G OUT: : {} {}" ,gx, g_out);
        smallvec![(x_id, gx)]
    };
    tr.push(crate::trace::Node { value: y, parents_id: smallvec![x_id], vjp: Some(Box::new(vjp)), is_param: false })
}
