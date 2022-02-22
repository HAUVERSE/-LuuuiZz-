
use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2, OwnedRepr, ArrayBase, Dim };
use linfa::Dataset;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::grid::Grid;
use plotlib::view::ContinuousView;
use plotlib::view::View;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;

type MyDataset = DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,>;

fn get_dataset() -> Dataset<f32, i32, ndarray::Dim<[usize;1]>> {
    let mut reader = Reader::from_path("./src/heart.csv").unwrap();

    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;

    let features = headers[0..target_index].to_vec();
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);

    return Dataset::new(records, targets).with_feature_names(features);
}

fn get_headers( reader: &mut Reader<File>) -> Vec<String> {
    return reader
        .headers().unwrap().iter()
        .map(|r| r.to_owned())
        .collect();
}

fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader
        .records()
        .map( |r|
              r.unwrap().iter()
              .map(|field| field.parse::<f32>().unwrap())
              .collect::<Vec<f32>>()
              )