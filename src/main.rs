/*
The core logic was written in the context of a coding sprint (a little over 3h.)
I then cleaned up the code, removing the .unwrap() that were written in for the sake
of testing the idea.

The idea was to implement a really simple trading idea, how much profit would we expect to make
assuming we were able to predict if a week from now, we'd make a profit.

Since it was a proof of concept, we only engineered features at a day level.
For QQQ using the free data from Investing.com and played with that.
The code is made such it would work with data taken from Investing.com.

    The goal is to create a AI signal tester. So we will need to train a model, and
    then use data to predict if entering right now would have generated a profit a week
    from now.

Currently for simplicity what we will do is always enter for exactly a week.
We will be using daily data.

We will take data and load it into a vector of Day structs.
Then we will run through the data to check if buying some action, say QQQ would generate profit.

We will be using a logistic regression as our model, nothing fancy.

General plan:

1. Load data

2. Do the feature engineering/analysis required in Rust.

3. Train the logistic regression on part of the data.

4. Run the algorithm on the rest of the data to see what would happen.
The goal is to see how much profit we would have made at the end.

Each part will be a function, and the main will really be calling those functions.
We won't be bothering with a GUI for now.
*/

// Imports
use colorize::AnsiColor;
use inquire::InquireError::{OperationCanceled, OperationInterrupted};
use inquire::Text;
use linfa::prelude::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, ArrayBase, Dim, OwnedRepr};
use std::{collections::VecDeque, process::ExitCode};

// Defining Row Struct
#[derive(Clone, Debug)]
struct Day {
    date: String,
    price: f64,
    open: f64,
    high: f64,
    low: f64,
    volatility: f64,
    percent_change: f64,
    is_bullish: bool,
    intraday_price_range: f64,
    price_spread_from_open: f64,
    make_profit_a_week_from_now: bool,
}

fn load_data(
    mut csv_path: String,
    flag_ignore_error: bool,
    is_header_present: bool,
    present_to_past: bool,
) -> Result<Vec<Day>, Box<csv::Error>> {
    let reader_builder = |csv_path: &str| {
        csv::ReaderBuilder::new()
            .has_headers(is_header_present)
            .from_path(csv_path)
    };

    let reader = loop {
        match reader_builder(&csv_path) {
            Ok(reader) => break reader,
            Err(err) => {
                eprintln!(
                    "{}",
                    format!(
                        "An error has occured as we were building the reader. \nPlease try again. Err: {err}"
                    )
                    .red()
                );

                csv_path = loop {
                    match Text::new("Please enter once again a csv path: ").prompt() {
                        Ok(csv_path) => break csv_path,
                        Err(OperationCanceled | OperationInterrupted) => {
                            println!("User has decided to terminate program. Buh-bye.");
                            std::process::exit(0);
                        }
                        Err(err) => eprintln!("An error has occured: {err}. Please try again!"),
                    }
                }
            }
        }
    };

    let mut csv_data: Vec<Day> = Vec::new();

    for (i, record) in reader.into_records().enumerate() {
        let row_number = i + 2;

        let record = match record {
            Ok(record) => record,
            Err(err) => {
                if flag_ignore_error {
                    continue;
                } else {
                    return Err(Box::new(err));
                }
            }
        };

        let parse_f64 = |s: &str, category: &str| -> f64 {
            s.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Failed to parse entry at row '{row_number}', in category {category} as f64. Defaulting to 0.0.");
                0.0 //default
            })
        };

        // If result, of unwrap_or_default is not stored, it goes out of scope
        // Could write it inline, but I opted for clarity.
        let raw_volatility = record[5].parse::<String>().unwrap_or_default();
        let volatility_str = raw_volatility.trim_end_matches('M');
        let volatility = parse_f64(volatility_str, "volatility");

        let raw_percent_change = record[6].parse::<String>().unwrap_or_default();
        let percent_change_str = raw_percent_change.trim_end_matches('%');
        let percent_change = parse_f64(percent_change_str, "percent_change");

        let row = Day {
            date: record[0].to_string(),
            price: parse_f64(&record[1], "price"),
            open: parse_f64(&record[2], "open"),
            high: parse_f64(&record[3], "high"),
            low: parse_f64(&record[4], "low"),
            volatility,
            percent_change,
            is_bullish: parse_f64(&record[1], "price") > parse_f64(&record[2], "open"),
            intraday_price_range: parse_f64(&record[3], "high") - parse_f64(&record[4], "low"),
            price_spread_from_open: parse_f64(&record[1], "price") - parse_f64(&record[2], "open"),
            make_profit_a_week_from_now: false, //default value
        };

        csv_data.push(row);
    }
    if present_to_past {
        return Ok(csv_data.iter().cloned().rev().collect());
    } else {
        return Ok(csv_data);
    }
}

fn feature_engineering(mut csv_data: Vec<Day>) -> Vec<Day> {
    //Since we defined our structs to contain the important information already, all we need to do is to keep a running
    // tally of if we would have made profit had we bought a week from now
    // a reasonable idea to start with is a sliding window
    // [a b c d e f g] when the sliding window is finally full
    // We would compare the price of the current date with that of the oldest entry in the sliding window
    // If current date is higher, then we'd have made profit
    // Else we would have lost money.

    let mut sliding_window: VecDeque<usize> = VecDeque::new();

    let indices: Vec<usize> = (0..csv_data.len()).rev().collect();

    for &i in &indices {
        if sliding_window.len() == 7 {
            if let Some(oldest_index) = sliding_window.pop_back() {
                csv_data[oldest_index].make_profit_a_week_from_now =
                    csv_data[i].price > csv_data[oldest_index].price;
            }
        }
        sliding_window.push_front(i);
    }

    let all_true = csv_data.iter().all(|day| day.make_profit_a_week_from_now);
    let all_false = csv_data.iter().all(|day| !day.make_profit_a_week_from_now);

    if all_true || all_false {
        println!("All entries in csv)data have the same class: {}", all_true);
        panic!("Too few classes: Cannot train logistic regression with only one label");
    }

    csv_data
}

fn train_logistic_regression(
    csv_data: Vec<Day>,
    train_split: f64,
) -> (
    ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) {
    // For this part, we take the easy way out
    // and make use of linfa.
    // We could have implemented it on our own, but no.
    let train_split_cutoff: usize = ((csv_data.len() as f64) * train_split) as usize;
    let train_data = &csv_data[..train_split_cutoff];
    let test_data = &csv_data[train_split_cutoff..];

    let x_train = match Array2::from_shape_vec(
        (train_data.len(), 9),
        train_data
            .iter()
            .flat_map(|d| {
                vec![
                    d.price,
                    d.open,
                    d.high,
                    d.low,
                    d.volatility,
                    d.percent_change,
                    if d.is_bullish { 1.0 } else { 0.0 },
                    d.intraday_price_range,
                    d.price_spread_from_open,
                ]
            })
            .collect(),
    ) {
        Ok(train_dataset) => train_dataset,
        Err(err) => {
            eprintln!(
                "{}",
                format!(
                    "An error occured as we were creating the training dataset\nError Code: {err}."
                )
                .red()
            );
            println!("{}", format!("Program will now terminate.").yellow());
            std::process::exit(-1);
        }
    };

    let y_train =
        Array1::from_iter(train_data.iter().map(
            |d| {
                if d.make_profit_a_week_from_now {
                    1
                } else {
                    0
                }
            },
        ));

    let model_training = LogisticRegression::default().fit(&Dataset::new(x_train, y_train));
    match model_training {
        Ok(model) => {
            if let Ok(x_test) = Array2::from_shape_vec(
                (test_data.len(), 9),
                test_data
                    .iter()
                    .flat_map(|d| {
                        vec![
                            d.price,
                            d.open,
                            d.high,
                            d.low,
                            d.volatility,
                            d.percent_change,
                            if d.is_bullish { 1.0 } else { 0.0 },
                            d.intraday_price_range,
                            d.price_spread_from_open,
                        ]
                    })
                    .collect(),
            ) {
                let y_test = Array1::from_iter(test_data.iter().map(|d| {
                    if d.make_profit_a_week_from_now {
                        1
                    } else {
                        0
                    }
                }));

                let predictions = model.predict(&x_test);
                let accuracy = match predictions.confusion_matrix(&y_test) {
                    Ok(confusion_matrix) => confusion_matrix.accuracy(),
                    Err(err) => {
                        eprintln!("An error occured as we tried accessing the evaluation metric (accuracy)\nError Code: {err}");
                        std::process::exit(-1);
                    }
                };

                println!("Model accuracy: {:.2}%", accuracy * 100.0);

                let x_test = match Array2::from_shape_vec(
                    (test_data.len(), 9),
                    test_data
                        .iter()
                        .flat_map(|d| {
                            vec![
                                d.price,
                                d.open,
                                d.high,
                                d.low,
                                d.volatility,
                                d.percent_change,
                                if d.is_bullish { 1.0 } else { 0.0 },
                                d.intraday_price_range,
                                d.price_spread_from_open,
                            ]
                        })
                        .collect(),
                ) {
                    Ok(dataset) => dataset,
                    Err(err) => {
                        println!("As we attempted to create the x test dataset, an error occurred. \nError Code: {err}");
                        println!("Please verify the dataset and try again, the program will now terminate.");
                        std::process::exit(-1)
                    }
                };

                let y_test = Array1::from_iter(test_data.iter().map(|d| {
                    if d.make_profit_a_week_from_now {
                        1
                    } else {
                        0
                    }
                }));

                let predictions = model.predict(&x_test);
                let all_1 = predictions.iter().all(|&entry| entry == 1);
                if all_1 {
                    println!("The prediction array predicts one for everything");
                }

                let accuracy = predictions.confusion_matrix(&y_test).unwrap().accuracy();
                println!("Model accuracy: {:.2}%", accuracy * 100.0);

                (predictions, x_test)
            } else {
                eprintln!("There has been an error in creating the test set.");
                (Array1::ones(1), Array2::ones((1, 1)))
            }
        }
        Err(model_fail) => panic!("Model training failed :{:?}", model_fail),
    }
}

fn main() -> ExitCode {
    // Taking in user input
    let csv_path = loop {
        match Text::new("Please pass in the path to your file: ").prompt() {
            Ok(csv_path) => break csv_path,
            Err(OperationCanceled | OperationInterrupted) => {
                println!("User has decided to interrupt program, program will now terminate.");
                break String::from("interrupted");
            }
            Err(error) => {
                eprintln!("Error in reading from command line: {error}. Please try again.")
            }
        }
    };

    if csv_path == "interrupted" {
        return ExitCode::FAILURE;
    }

    // Defining our variables
    let has_headers = true;
    let ignore_errors = true;
    let present_to_past = true;
    let mut portfolio_size = 100000.0;
    let default_allocation = 0.1; //10%
    let mut shares_owned = 0.0; //assume fractional shares are allowed
    let mut invested = false;
    let mut days_since_entered_position = 0;
    let mut purchases_count = 0;
    let mut sell_count = 0;

    // We got a path so now we go to loading
    let rows = load_data(csv_path, has_headers, ignore_errors, present_to_past);
    if let Ok(ref csv_data) = rows {
        let rows_with_label = feature_engineering(csv_data.to_vec());
        let (predictions, x_test) = train_logistic_regression(rows_with_label, 0.8);

        // Trading Logic
        for (price, prediction) in x_test.column(0).iter().zip(predictions.iter()) {
            let would_make_profit = *prediction == 1;
            if invested {
                if would_make_profit && days_since_entered_position == 7 {
                    // We maintain position and reset timer
                    days_since_entered_position = 0;
                } else if days_since_entered_position >= 7 {
                    // We liquidate holdings
                    sell_count += 1;
                    days_since_entered_position = 0;
                    portfolio_size += shares_owned * price;
                    shares_owned = 0.0;
                    invested = false;
                } else {
                    days_since_entered_position += 1;
                }
            } else if would_make_profit {
                let allocation = price * default_allocation;
                shares_owned = allocation / price;

                purchases_count += 1;
                portfolio_size -= allocation;

                invested = true;
            }
        }
    } else {
        // This mean we didn't manage to load the data
        return ExitCode::FAILURE;
    }

    println!("Final portfolio value is {:.2}", portfolio_size);
    println!("Shares owned at the end of experiment is {}", shares_owned);
    // Check number of shares at the end
    match rows {
        Ok(ref data) if !data.is_empty() => {
            let final_price = data[data.len() - 1].price;
            println!(
                "Value of shares at the end of the experiment is {:.2}",
                shares_owned * final_price
            );
        }
        _ => {
            println!("Value of shares at the end of the experiment is -1.00 (rows is not defined somehow)");
        }
    }
    println!(
        "Total number of transactions is: {}",
        sell_count + purchases_count
    );
    println!("Number of purchases is: {}", purchases_count);
    println!("Number of sells is: {}", sell_count);

    return ExitCode::SUCCESS;
}
