use crate::tensor::*;
use crate::ops::*;
use smallvec::SmallVec;

pub struct NodeId(pub usize); 




pub enum Node{
    Leaf{
        tensor : Tensor, 
        requires_grad: bool
    },
    InternalNode{
        op: Op, 
        parents: SmallVec<[NodeId; 2]>,
        tensor: Tensor
    }
}

pub struct Graph{
    nodes : Vec<Node>
}

impl Graph{

    #[inline]
    pub fn push(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }
    /*sets requires_grad to false since just an input */
    pub fn input(& mut self, t: Tensor) -> NodeId{
        self.push(Node::Leaf{tensor: t, requires_grad: false})
    }
    pub fn param(&mut self, t: Tensor) -> NodeId{
        self.push(Node::Leaf { tensor: t, requires_grad: true })
    }
}
    