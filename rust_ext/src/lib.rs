use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Sum all elements of a 1-D f64 array.
#[pyfunction]
fn array_sum(py: Python<'_>, arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(arr.as_array().iter().sum())
}

/// Matrix dot product of two 2-D f64 arrays.
#[pyfunction]
fn matrix_dot<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    let n = a.shape()[0];
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[[i, k]];
            for j in 0..n {
                result[[i, j]] += a_ik * b[[k, j]];
            }
        }
    }
    Ok(result.into_pyarray(py))
}

/// Element-wise square root of a 1-D f64 array.
#[pyfunction]
fn elementwise_sqrt<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result: Vec<f64> = arr.as_array().iter().map(|x| x.sqrt()).collect();
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array_sum, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_dot, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise_sqrt, m)?)?;
    Ok(())
}
