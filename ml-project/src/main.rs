
use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2, OwnedRepr, ArrayBase, Dim };
use linfa::Dataset;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::grid::Grid;
use plotlib::view::ContinuousView;