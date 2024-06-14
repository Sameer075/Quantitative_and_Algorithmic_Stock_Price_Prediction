document.addEventListener('DOMContentLoaded', function() {
    const forecastButton = document.getElementById('forecastButton');
    forecastButton.addEventListener('click', function() {
        // Change button text to loading and disable it
        forecastButton.textContent = 'Loading...';
        forecastButton.disabled = true;

        const stockSymbol = document.getElementById('stockSymbol').value;
        const modelType = forecastButton.getAttribute('data-model'); // Get the model type from the button attribute

        let url;
        if (modelType === 'linear') {
            url = '/forecast/';
        } else if (modelType === 'random_forest') {
            url = '/forecast/';
        } else if (modelType === 'ensemble') {
            url = '/forecast/';
        } else if (modelType === 'intraday') {
            url = '/forecast/';
        }

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({
                stockSymbol: stockSymbol,
                modelType: modelType
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Data received:', data);
            // Revert button text back to result and enable it
            forecastButton.textContent = 'Fetched Result';
            forecastButton.disabled = false;
            if (data.error) {
                console.error('Error:', data.error);
            } else {
                if (data.candleData) {
                    updateCandleChart(data.candleData, stockSymbol);
                } else {
                    console.error('Error: Candlestick data is missing.');
                }
                if (data.forecastData) {
                    updateLineChart(data.forecastData);
                    showEvaluationMetrics(data.forecastData.metrics);
                } else {
                    console.error('Error: Forecast data is missing.');
                }
            }
        })
        .catch(error => console.error('Error:', error));
        // Revert button text back to result and enable it
        forecastButton.textContent = 'Fetching Result';
        forecastButton.disabled = false;
    });
});

function updateCandleChart(data, stockSymbol) {
    const dates = data.map(item => new Date(item.x));
    const opens = data.map(item => item.y[0]);
    const highs = data.map(item => item.y[1]);
    const lows = data.map(item => item.y[2]);
    const closes = data.map(item => item.y[3]);

    // Calculate the range for y-axis
    const minPrice = Math.min(...lows);
    const maxPrice = Math.max(...highs);
    const buffer = 10; // Adjust as needed
    const windowHeight = 500; // Adjust as needed

    const layout = {
        title: {
            text: `${stockSymbol} Stock Price`,
            font: {
                color: 'black' // Set title text color to black for light theme
            }
        },
        xaxis: {
            title: {
                text: 'Date',
                font: {
                    color: 'black' // Set x-axis label text color to black for light theme
                }
            },
            rangeslider: { visible: false },
            tickfont: {
                color: 'black' // Set x-axis tick labels color to black for light theme
            }
        },
        yaxis: {
            title: {
                text: 'Price',
                standoff: 20,
                font: {
                    color: 'black' // Set y-axis label text color to black for light theme
                }
            },
            tickfont: {
                color: 'black' // Set y-axis tick labels color to black for light theme
            },
            range: [minPrice - buffer, maxPrice + buffer]
        },
        height: windowHeight,
        margin: {
            t: 100,
            b: 100,
            l: 100,
            r: 50,
            pad: 10
        },
        paper_bgcolor: 'white', // Set background color to white for light theme
        plot_bgcolor: 'rgba(0, 0, 0, 0.03)', // Set plot area background color for light theme
        showlegend: false,
        border: {
            color: 'black', // Set border color to black for light theme
            width: 1 // Reduce border width for light theme
        }
    };

    const candlestickData = [{
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        type: 'candlestick',
        increasing: { line: { color: 'green', width: 2 } },
        decreasing: { line: { color: 'red', width: 2 } }
    }];

    Plotly.newPlot('candlestickChart', candlestickData, layout);
}

function updateLineChart(forecastData) {
    const dates = forecastData.forecast_dates;
    const actualClose = forecastData.actual_close;
    const predictedClose = forecastData.predicted_close;

    const actualTrace = {
        x: dates,
        y: actualClose,
        mode: 'lines',
        line: {
            color: 'blue',
            width: 2
        },
        name: 'Actual Close Price'
    };

    const predictedTrace = {
        x: dates,
        y: predictedClose,
        mode: 'lines',
        line: {
            color: 'red',
            width: 2
        },
        name: 'Predicted Close Price'
    };

    const layout = {
        title: {
            text: 'Actual vs Predicted Close Prices',
            font: {
                color: 'black' // Set title text color to black for light theme
            }
        },
        xaxis: {
            title: {
                text: 'Date',
                font: {
                    color: 'black' // Set x-axis label text color to black for light theme
                }
            },
            tickfont: {
                color: 'black' // Set x-axis tick labels color to black for light theme
            }
        },
        yaxis: {
            title: {
                text: 'Close Price',
                standoff: 20,
                font: {
                    color: 'black' // Set y-axis label text color to black for light theme
                }
            },
            tickfont: {
                color: 'black' // Set y-axis tick labels color to black for light theme
            }
        },
        height: 500, // Adjust the height to match the candlestick chart
        margin: {
            t: 100, // Adjust top margin to give space for title
            b: 100, // Adjust bottom margin for x-axis label
            l: 100, // Adjust left margin for y-axis label
            r: 50, // Adjust right margin for better visibility
            pad: 2 // Padding
        },
        paper_bgcolor: 'white', // Set background color to white for light theme
        plot_bgcolor: 'rgba(0, 0, 0, 0.03)', // Set plot area background color for light theme
        showlegend: true, // Show legend
        legend: {
            font: {
                color: 'black' // Set legend text color to black for light theme
            }
        }
    };

    try {
        Plotly.newPlot('forecastChart', [actualTrace, predictedTrace], layout);
    } catch (error) {
        console.error('Error creating Plotly chart:', error);
    }
}

function showEvaluationMetrics(metrics) {
    const evaluationMetricsDiv = document.getElementById('evaluationMetrics');
    evaluationMetricsDiv.innerHTML = `
    <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: #e6e6e6;">
                <td style="border: 1px solid #ddd; padding: 8px;">Mean Absolute Error</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${metrics['Mean Absolute Error'].toFixed(2)}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Mean Squared Error</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${metrics['Mean Squared Error'].toFixed(2)}</td>
            </tr>
            <tr style="background-color: #e6e6e6;">
                <td style="border: 1px solid #ddd; padding: 8px;">Root Mean Squared Error</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${metrics['Root Mean Squared Error'].toFixed(2)}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">R-squared</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${metrics['R-squared'].toFixed(2)}</td>
            </tr>
        </tbody>
    </table>
    `;

    // Pie chart data
    const pieData = [{
        labels: ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        values: [metrics['Mean Absolute Error'], metrics['Mean Squared Error'], metrics['Root Mean Squared Error']],
        type: 'pie'
    }];

    // Pie chart layout
    const pieLayout = {
        title: {
            text: 'Proportion of Errors',
            font: {
                size: 24
            }
        },
        showlegend: true
    };

    // Create pie chart
    Plotly.newPlot('pieChart', pieData, pieLayout, { displayModeBar: false });

    // Scatter plot data for R-squared
    const scatterData = [{
        x: ['R-squared'],
        y: [metrics['R-squared']],
        mode: 'markers',
        marker: {
            size: 15,
            color: 'orange'
        },
        type: 'scatter',
        name: 'R-squared'
    }];

    // Scatter plot layout
    const scatterLayout = {
        title: {
            text: 'R-squared Scatter Plot',
            font: {
                size: 24
            }
        },
        xaxis: {
            title: 'Metric',
            titlefont: {
                size: 18
            }
        },
        yaxis: {
            title: 'Value',
            titlefont: {
                size: 18
            }
        }
    };

    // Create scatter plot
    Plotly.newPlot('scatterContainer', scatterData, scatterLayout, { displayModeBar: false });
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}