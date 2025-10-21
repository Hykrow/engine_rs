use crate::ops::OpType;
use std::{ops, string}; 
use std::rc::Rc; 
use std::cell::RefCell; 
use std::ops::Add;

#[derive(Clone)]
pub struct ValueRef(pub Rc<RefCell<Value>>);


pub struct Value{
    pub val : f32, 
    pub op : OpType, 
    pub grad : f32, 
    pub parents: Vec<ValueRef>, 
    pub label: Option<String>, 
}

impl ValueRef{
    pub fn leaf(val : f32, label: Option<String>) -> ValueRef{
        Rc::new(RefCell::new( Value {
            val, 
            op: OpType::None, 
            grad: 0.0, 
            parents :vec![], 
            label,

        }))
    }


    pub fn from_op(op: OpType, parents: Vec<ValueRef>, label: Option<String>)-> ValueRef{
        op.apply(parents)
    }

}


impl Add for ValueRef{
    type Output = Self; 

    fn add (self, rhs: Self) -> Self::Output{
        Self{
            ValueRef
            
        }
    }
}