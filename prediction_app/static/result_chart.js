document.addEventListener('DOMContentLoaded', function() {
    const resultButton = document.getElementById('resultButton');
    resultButton.addEventListener('click', function() {
        // Change button text to loading and disable it
        resultButton.textContent = 'Loading...';
        resultButton.disabled = true;
        
        const stockSymbol = document.getElementById('stockSymbol').value;

        let url = '/result/';

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({
                stockSymbol: stockSymbol,
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Data received:', data);
            // Revert button text back to result and enable it
            resultButton.textContent = 'Fetched Result';
            resultButton.disabled = false;
            if (data.error) {
                console.error('Error:', data.error);
            } else {
                if (data.candleData) {
                    updateCandleChart(data.candleData, stockSymbol);
                } else {
                    console.error('Error: Candlestick data is missing.');
                }
                if (data.resultData) {
                    showEvaluationMetrics(data.resultData.metrics);
                    plotConfusionMatrixHeatmap(data.resultData.metrics.confusionMatrix);
                    plotROCCurve(data.resultData.metrics.roc);
                } else {
                    console.error('Error: Result data is missing.');
                }
            }
        })
        .catch(error => console.error('Error:', error));
        // Revert button text back to result and enable it
        resultButton.textContent = 'Fetching Result';
        resultButton.disabled = false;
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


function showEvaluationMetrics(metrics) {
    const evaluationMetricsDiv = document.getElementById('evaluationMetrics');
    evaluationMetricsDiv.innerHTML = `
    <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background-color: #f8f9fa;">
                <th style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">Metric</th>
                <th style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">Value</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: #ffffff;">
                <td style="border: 1px solid #dee2e6; padding: 8px;">Accuracy</td>
                <td style="border: 1px solid #dee2e6; padding: 8px;">${metrics['Accuracy'].toFixed(5)}</td>
            </tr>
            <tr style="background-color: #f8f9fa;">
                <td style="border: 1px solid #dee2e6; padding: 8px;">Precision</td>
                <td style="border: 1px solid #dee2e6; padding: 8px;">${metrics['Precision'].toFixed(5)}</td>
            </tr>
            <tr style="background-color: #ffffff;">
                <td style="border: 1px solid #dee2e6; padding: 8px;">Recall</td>
                <td style="border: 1px solid #dee2e6; padding: 8px;">${metrics['Recall'].toFixed(5)}</td>
            </tr>
            <tr style="background-color: #f8f9fa;">
                <td style="border: 1px solid #dee2e6; padding: 8px;">F1-Score</td>
                <td style="border: 1px solid #dee2e6; padding: 8px;">${metrics['F1-Score'].toFixed(5)}</td>
            </tr>
        </tbody>
    </table>
    `;

    // Bar chart data
    const barData = [{
        x: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y: [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']],
        type: 'bar',
        marker: {
            color: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            opacity: 0.8
        }
    }];

    // Bar chart layout
    const barLayout = {
        title: {
            text: 'Model Performance Metrics',
            font: {
                size: 24
            }
        },
        xaxis: {
            title: {
                text: 'Metrics',
                font: {
                    size: 18
                }
            }
        },
        yaxis: {
            title: {
                text: 'Score',
                font: {
                    size: 18
                }
            },
            range: [0, 1] // assuming the metrics are within the range 0 to 1
        },
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff'
    };

    // Create bar chart
    Plotly.newPlot('barChart', barData, barLayout, { displayModeBar: true });
}

function plotConfusionMatrixHeatmap(cm) {
    const labels = ['Up Day', 'Down Day']; // Labels for confusion matrix

    const data = [{
        z: cm,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true,
        colorbar: {
            title: 'Count',
            titlefont: {
                size: 14
            }
        },
        zmin: 0,
        zmax: Math.max(...cm.flat()) // Automatically adjusts the color scale based on the maximum value in the confusion matrix
    }];

    const annotations = [];
    for (let i = 0; i < cm.length; i++) {
        for (let j = 0; j < cm[i].length; j++) {
            annotations.push({
                x: labels[j],
                y: labels[i],
                text: cm[i][j],
                font: {
                    size: 15,
                    color: 'black'
                },
                showarrow: false
            });
        }
    }

    const layout = {
        title: 'Confusion Matrix',
        xaxis: {
            title: 'Predicted',
            tickfont: {
                size: 14 // Adjusts the font size of the x-axis labels
            }
        },
        yaxis: {
            title: 'Actual',
            tickfont: {
                size: 14 // Adjusts the font size of the y-axis labels
            }
        },
        annotations: annotations,
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('confusionMatrixHeatmap', data, layout);
}

function plotROCCurve(rocData) {
    const fpr = rocData.fpr;
    const tpr = rocData.tpr;
    const rocAuc = rocData.auc;

    const trace = {
        x: fpr,
        y: tpr,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'blue' },
        name: `ROC curve (area = ${rocAuc.toFixed(2)})`
    };

    const diagonal = {
        x: [0, 1],
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        line: { dash: 'dash', color: 'gray' },
        name: 'Random guess'
    };

    const layout = {
        title: 'Receiver Operating Characteristic (ROC) Curve',
        xaxis: { 
            title: 'False Positive Rate',
            tickfont: {
                size: 14
            }
        },
        yaxis: { 
            title: 'True Positive Rate',
            tickfont: {
                size: 14
            }
        },
        showlegend: true,
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('rocCurve', [trace, diagonal], layout);
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