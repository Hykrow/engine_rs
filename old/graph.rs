use crate::tensor::*;
use crate::ops::*;
use smallvec::SmallVec;
use smallvec::ToSmallVec;
use std::collections::HashMap;


pub enum Optimizer{
    Adam
}

// TODO: changer pour que ce soit juste Node, car la pas très clean
#[derive(Clone, Copy)]
pub struct Var { pub id: usize }
//TODO: rajouter l'id du Graphe / tape comme ca on peut jsute add 2 nodes sans avoir le contexte du tape. 

pub enum Node{
    Leaf{
        tensor : Tensor, 
        requires_grad: bool
    },
    InternalNode{
        op: Op, 
        parents: SmallVec<[Var; 2]>,
        tensor: Tensor
    }
}

pub struct Graph{
    nodes : Vec<Node>, 
    // TODO ? ajouter un truc root qui pointe vers le dernier élément ajouté.  ?
}

impl Graph{
    pub fn new() -> Graph{
        let g :Vec<Node> = Vec::new(); 
        Graph{nodes: g}
    }
    pub fn value(&self, var: Var) -> &Tensor {
        match &self.nodes[var.id] {
            Node::Leaf { tensor, .. } => tensor,
            Node::InternalNode { tensor, .. } => tensor,
        }
    }
    #[inline]
    pub fn push(&mut self, node: Node) -> Var{
        let id = self.nodes.len();
        self.nodes.push(node);
        Var{id}
    }
    /*sets requires_grad to false since just an input */
    pub fn input(& mut self, t: Tensor) -> Var{
        self.push(Node::Leaf{tensor: t, requires_grad: false})
    }
    pub fn param(&mut self, t: Tensor) -> Var{
        self.push(Node::Leaf { tensor: t, requires_grad: true })
    }

    pub fn dfs(&self, visited: & mut Vec<bool>, node_order: & mut Vec<Var>, node : Var){
        if visited[node.id] {
            return; 
        }else{
            visited[node.id]  = true;
            match &self.nodes[node.id]{
                Node::Leaf { .. }=>{}, 
                Node::InternalNode{op: _op, parents, tensor: _tensor}=>{
                    for &p in parents{
                        self.dfs(visited, node_order, p);
                    }
                }
            }
           node_order.push(node);

        }
    }
    pub fn topological_sort(&self, root: Var) -> Vec<Var>{
        let mut visited = vec![false; self.nodes.len()]; 
        let mut node_order = Vec::new(); 
        self.dfs(& mut visited, & mut node_order, root);
        node_order
    }

    pub fn compute_gradients(&self, root: Var) -> HashMap<usize, Tensor>{
        let order = self.topological_sort(root);
        //TODO : compute le gradient en amont avec la root (assumes MSE)
        let mut tmp : Vec<Option<Tensor>> = vec![None; self.nodes.len()];
        tmp[root.id] = Some(Tensor::ones(&self.value(root).shape));

        let mut grads :HashMap<usize, Tensor> = HashMap::new();


        for &node in order.iter().rev(){ // check si il faut un reverse? 

            let before_grad = match tmp[node.id].as_ref() {
                Some(g) => g,
                None => continue, // rien à propager
            };            
            match &self.nodes[node.id]{
                Node::Leaf { requires_grad , ..} =>{
                    if *requires_grad{
                        grads.insert(node.id, before_grad.clone());
                    }
                }, 
                Node::InternalNode { op, parents, tensor } =>{
                    let parents_tensors : Vec<&Tensor> = parents.iter().map(|&va| self.value(va)).collect();
                    let locals_grad = op.backward(&parents_tensors, tensor, before_grad); // on recupere le gradiant pour les enfants.

                    for(pid, to_accum) in locals_grad.into_iter().enumerate(){
                        let va = parents[pid];
                        accum(&mut tmp[va.id], to_accum); 
                    }


                }
            }
        
            
        }
        grads
    }
    pub fn apply(&mut self, op: Op, parents: &[Var]) -> Var{
        let parent_tensors : Vec<&Tensor> = parents.iter().map(|&v| self.value(v)).collect();
        let out = op.forward(&parent_tensors);
        self.push(Node::InternalNode { op: op, parents: parents.to_smallvec(), tensor: out })    
        
    }

}
    

pub fn accum( slot: &mut Option<Tensor>, tensor: Tensor ){
    *slot=Some(match slot.take(){
        None => tensor,
        Some(g)=> &g + &tensor
    })
}
