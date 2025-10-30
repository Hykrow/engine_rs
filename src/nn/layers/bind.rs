use crate::trace::NodeId; 

pub struct ParamCursor<'a>{
    p: &'a [NodeId], 
    i: usize, 
}

impl <'a>ParamCursor<'a>{
    pub fn new (p: &'a [NodeId]) -> Self{Self {p, i:0}}

    pub fn take(&mut self) -> NodeId{
        let id = self.p[self.i]; 
        self.i+=1; 
        id
    }

    pub fn take2(&mut self) -> (NodeId, NodeId){
        (self.take(), self.take())
    }

    pub fn remaining(&self) -> usize{
        self.p.len() - self.i
    }

}
