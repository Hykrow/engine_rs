use crate::tensor::*;
use crate::ops::*;
use smallvec::SmallVec;

pub struct NodeId(pub usize); 


struct Node{
    op: Op, 
    parents: SmallVec<[NodeId; 2]>,
    tensor: Tensor
}

pub struct Graph{
    nodes : Vec<Node>
}

impl Graph{
    #[inline]
    fn push(&mut self, op: Op, parents: SmallVec<[NodeId; 2]>, tensor: Tensor) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, parents, tensor });
        id
    }
    pub fn input(& mut self, t: Tensor) -> NodeId{
        self.push(Op::Leaf, SmallVec::new(), t)
    }
}