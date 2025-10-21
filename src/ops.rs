#[derive(Clone, Copy, Debug)]
pub enum OpType { None, Add, Mul, ReLU, Tanh }
use crate::value::ValueRef;
impl OpType {
    /// Arite attendue (utile pour asserts/debug)
    pub fn arity(self) -> usize {
        match self {
            OpType::None => 1, // feuille: on propage juste la valeur
            OpType::Add | OpType::Mul => 2,
            OpType::ReLU | OpType::Tanh => 1,
        }
    }

    /// Calcule la valeur de sortie à partir des valeurs des parents
    pub fn apply(self, parents: Vec<ValueRef>) -> ValueRef {

        let xs = parents.iter().map(|p| p.borrow.val()).collect();

        debug_assert_eq!(xs.len(), self.arity(), "Arity mismatch for {:?}", self);
        match self {
            OpType::None => xs[0],
            OpType::Add  => xs[0] + xs[1],
            OpType::Mul  => xs[0] * xs[1],
            OpType::ReLU => xs[0].max(0.0),
            OpType::Tanh => xs[0].tanh(),
        }
    }

    /// Gradients locaux d(out)/d(x_i) pour chaque parent i
    /// `out_val` peut simplifier certains cas (ex: tanh: 1 - tanh^2)
    pub fn local_grads(self, xs: &[f32], out_val: f32) -> Vec<f32> {
        debug_assert_eq!(xs.len(), self.arity(), "Arity mismatch for {:?}", self);
        match self {
            OpType::None => vec![1.0],
            OpType::Add  => vec![1.0, 1.0],
            OpType::Mul  => vec![xs[1], xs[0]],
            OpType::ReLU => vec![(xs[0] > 0.0) as i32 as f32],
            OpType::Tanh => {
                // d tanh / dx = 1 - tanh(x)^2 = 1 - out^2
                vec![1.0 - out_val.powi(2)]
            }
        }
    }
}
