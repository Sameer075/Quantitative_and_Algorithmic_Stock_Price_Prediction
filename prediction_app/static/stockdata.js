/*
// Constants  2ZYF5RPKRKDZROJE,  YLPNUW1V9XP866L4,  LY5FCAMAD0T9VE5X
const API_KEY = 'O3XAP0V0U3MIC0FS';
const TIME_SERIES_DAILY = 'TIME_SERIES_DAILY';

// Function to fetch stock data from Alpha Vantage API
async function fetchStockData(stockCode, timeSeriesType) {
    const url = `https://www.alphavantage.co/query?function=${timeSeriesType}&symbol=${stockCode}&apikey=${API_KEY}`;
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`Error fetching ${timeSeriesType} stock data:`, error);
        return null;
    }
}

// Function to update line chart using Chart.js
function updateLineChart(data) {
    const dailyData = data['Time Series (Daily)'];
    const dates = Object.keys(dailyData).reverse();
    const priceData = dates.map(date => {
        return {
            t: new Date(date),
            y: parseFloat(dailyData[date]['4. close'])
        };
    });

    const ctx = document.getElementById('stock-price-plot').getContext('2d');
    const stockPriceChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Stock Price',
                data: priceData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1,
                pointRadius: 0,
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                }],
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: 'Price'
                    }
                }]
            }
        }
    });
}

// Event listener for time range buttons
['1week', '1month', '1yr', '5yr', '10yr', 'max'].forEach(interval => {
    document.getElementById(`${interval}-btn`).addEventListener('click', async () => {
        const stockCode = document.getElementById('stock-search').value;
        const stockData = await fetchStockData(stockCode, TIME_SERIES_DAILY);
        if (stockData) {
            updateLineChart(stockData);
        }
    });
});

// Event listener for stock search button click
document.getElementById('stock-price-button').addEventListener('click', async () => {
    const stockCode = document.getElementById('stock-search').value.trim();
    if (stockCode.length > 0) {
        const stockData = await fetchStockData(stockCode, TIME_SERIES_DAILY);
        if (stockData) {
            updateLineChart(stockData);
        }
    }
});
*/