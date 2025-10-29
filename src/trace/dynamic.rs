use crate::tensor::Tensor;

use smallvec::SmallVec;



pub type NodeId = usize; 

type VjpFn = 
    Box<dyn Fn(&Tensor) -> SmallVec<[(NodeId, Tensor); 2]> +Send + Sync>;

pub struct Node{ // TODO: remplacer value par rien, pour du 100% JAX. la entre pytorch et jax..  

    pub value: Tensor, 

    pub parents_id: SmallVec<[NodeId; 2]>, 

    pub vjp: Option<VjpFn>,

    pub is_param: bool 
}

pub struct Trace{
    nodes: Vec<Node>, 

    params_id: Vec<NodeId>,
}

impl Trace{
    pub fn new() -> Trace 
    {
        Trace { nodes: Vec::new(), params_id: Vec::new() }
    }
    
    pub fn len(&self) -> usize
    {
        self.nodes.len()
    }    

    pub fn get_tensor(&self, id: NodeId)-> &Tensor
    {
        &self.nodes[id].value
    }

    pub fn push(&mut self, node: Node) -> NodeId
    {
        let id = self.len();
        self.nodes.push(node); 
        //if node.is_param{
        //    self.params_id.push(id);
        //}
        id
    }

    pub fn input(&mut self, t: Tensor) -> NodeId
    {
        self.push(Node{
            value: t, 
            parents_id: SmallVec::new(), 
            vjp: None, 
            is_param: false
        })
    }

    pub fn param(&mut self, t: Tensor) -> NodeId
    {
        let id = self.push(Node{
            value: t, 
            parents_id: SmallVec::new(), 
            vjp: None, 
            is_param: false
        }); 
        self.params_id.push(id); 
        id
    }
    
    pub fn order(&self, root: NodeId) -> Vec<NodeId>
    {
        let mut order: Vec<NodeId> = Vec::with_capacity(self.len());
        let mut visited = vec![false; self.len()];

        fn dfs(tr: &Trace, visited: &mut Vec<bool>, order: &mut Vec<NodeId>, u: NodeId)
        {
            if visited[u]{
                return;
            }
            visited[u]= true; 
            for &p in &tr.nodes[u].parents_id{
                dfs(tr, visited, order, p);
            }
            order.push(u);
        }
        dfs(&self, &mut visited, &mut order, root);
        order.reverse();
        order

    }
    
    pub fn accum(slot: &mut Option<Tensor>, delta: Tensor)
    {
        *slot =Some(
            match slot.take(){
                Some(g) => &g+&delta,
                None =>delta
            }
        )
    }

    pub fn backward_param_grads(&self, root: NodeId) -> Vec<Tensor>
    {
        let order = self.order(root);

        let mut grads: Vec<Option<Tensor>> = vec![None; self.len()];
        grads[root] = Some(Tensor::ones(&self.get_tensor(root).shape));

        for &node_id in order.iter(){
            let Some(ref g_out) = grads[node_id] else {println!("gradient non trouvé"); continue}; // permet de chopper le g_out dans le Some directement. Normalement déjà dedans car gradient déjà calculé par l'ordre topologique 
            if let Some(ref vjp) = self.nodes[node_id].vjp{ // si il a une vector jacobian product
                for (parent_id, tensor) in vjp(g_out){
                    Trace::accum(&mut grads[parent_id], tensor);
                }
            }
        }

        //TODO : faire map + iter pcq la c pas fou fou 
        let mut res = Vec::with_capacity(self.params_id.len());
        for &id in &self.params_id{
            res.push(grads[id].as_ref().unwrap_or_else(|| panic!("dynamic: le gradient d'un param n'est pas trouvable")).clone());
        }
        res

    }
    
}