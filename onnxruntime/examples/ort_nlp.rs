use onnxruntime::environment::Environment;
use onnxruntime::ndarray::prelude::*;
use onnxruntime::ndarray::{Data, IxDyn, OwnedRepr, RawData, ViewRepr};
use onnxruntime::{session::Session, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel};
use std::ops::Deref;
use std::time::Instant;

fn run<'s, 'm, 't>(
    session: &'s mut Session,
    input_ids: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    past_key_values: &[ArrayBase<OwnedRepr<f32>, IxDyn>],
) -> (i64, Vec<OrtOwnedTensor<'m, 't, f32, IxDyn>>)
where
    'm: 't,
    's: 'm,
{
    session.feed(input_ids).unwrap();
    past_key_values.into_iter().for_each(|past_key_value| {
        session.feed(*past_key_value).unwrap();
    });
    session.feed(attention_mask).unwrap();
    // let start = Instant::now();
    session.inner_run().unwrap();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());
    let new_id = {
        let logits: OrtOwnedTensor<'_, '_, f32, IxDyn> = session.read().unwrap();
        let new_id = argmax(logits.deref(), 0) as i64;
        new_id
    };
    // let start = Instant::now();
    let out_past_key_values = session.read_vec(24).unwrap();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());

    (new_id, out_past_key_values)
}
fn run_second<'s, 'm, 't>(
    session: &'s mut Session,
    input_ids: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    past_key_values: &[Array<ViewRepr<f32>, IxDyn>],
) -> (i64, Vec<OrtOwnedTensor<'m, 't, f32, IxDyn>>)
where
    'm: 't,
    's: 'm,
{
    session.feed(input_ids).unwrap();
    past_key_values.into_iter().for_each(|past_key_value| {
        session.feed(*past_key_value).unwrap();
    });
    session.feed(attention_mask).unwrap();
    // let start = Instant::now();
    session.inner_run().unwrap();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());
    let new_id = {
        let logits: OrtOwnedTensor<'_, '_, f32, IxDyn> = session.read().unwrap();
        let new_id = argmax(logits.deref(), 0) as i64;
        new_id
    };
    // let start = Instant::now();
    let out_past_key_values = session.read_vec(24).unwrap();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());

    (new_id, out_past_key_values)
}

fn argmax<T>(matrix: &ArrayBase<ViewRepr<T>, IxDyn>, axis: usize) -> usize
where
    T: std::cmp::PartialOrd + Copy + std::fmt::Debug,
    ViewRepr<T>: RawData + Data,
    <ViewRepr<T> as RawData>::Elem: std::cmp::PartialOrd + Copy,
{
    for (_, row) in matrix.axis_iter(Axis(axis)).enumerate() {
        let (max_idx, _) =
            row.iter()
                .enumerate()
                .fold((0, row[[0, 0]]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        return max_idx;
    }
    return 100000;
}

fn generate(tokens: usize, session: &mut Session) {
    let mut input_ids = array![[0i64]];
    let mut attention_mask = vec![1i64; 3];
    let mut past_key_values: Vec<_> = (0..24)
        .map(|_| Array::<f32, _>::zeros(IxDyn(&[1, 12, 2, 64])))
        .collect();

    let (new_id, out_past_key_values) = run(
        session,
        input_ids,
        Array::from_shape_vec((1, attention_mask.len()), attention_mask.clone()).unwrap(),
        &past_key_values,
    );
    for _ in 0..tokens - 1 {
        // println!("i {:?}", i);

        // println!("input_ids {:?}", input_ids.shape());
        // println!("attention_mask {:?}", attention_mask.shape());
        // println!("past_key_values {:?}", past_key_values[0].shape());
        let (new_id, out_past_key_values) = run(
            session,
            input_ids,
            Array::from_shape_vec((1, attention_mask.len()), attention_mask.clone()).unwrap(),
            out_past_key_values.deref(),
        );
        input_ids = array![[new_id]];
        attention_mask.push(1);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        // .with_number_threads(1)?
        .with_model_from_file("gpt2.onnx")?;

    let start = Instant::now();
    generate(47, &mut session);
    println!(
        "Time elapsed in expensive_function() is: {:?}",
        start.elapsed()
    );
    Ok(())
}
