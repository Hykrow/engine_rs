use crate::tensor::Tensor;
use crate::graph::*;

#[derive(Clone)]
pub struct Param{
    pub value : Tensor, 
    pub last_var : Option<Var>
}

impl Param{
    pub fn new(value: Tensor) -> Param{
        Param{
            value, 
            last_var: None
        }
    }
    pub fn bind(&mut self, g: &mut Graph) -> Var{
        let v = g.param(self.value.clone());
        self.last_var = Some(v);
        v
    }
}
