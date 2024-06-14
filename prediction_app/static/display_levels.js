document.addEventListener('DOMContentLoaded', function() {
    const levelButton = document.getElementById('levelButton');
    levelButton.addEventListener('click', function() {
        const stockSymbol = document.getElementById('stockSymbol').value;

        let url = '/levels/';

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
            if (data.error) {
                console.error('Error:', data.error);
            } else {
                if (data.candleData) {
                    updateCandleChart(data.candleData, stockSymbol);
                } else {
                    console.error('Error: Candlestick data is missing.');
                }
                if (data.levelData) {
                    // Extract data from the response
                    const prices = data.levelData.prices; // Access prices directly from levelData
                    const counts = data.levelData.counts; // Access counts directly from levelData
                    const positiveLevels = data.levelData.positive_levels;
                    const negativeLevels = data.levelData.negative_levels;
                    const vah = data.levelData.VAH_VAL.VAH; // Access VAH value
                    const val = data.levelData.VAH_VAL.VAL; // Access VAL value

                    // Plot Positive Gaussian Curve
                    plotGaussianCurve(positiveLevels, prices, counts, 'Positive Gaussian Curve Levels with TPO', 'positiveCurve', vah, val);

                    // Plot Negative Gaussian Curve
                    plotGaussianCurve(negativeLevels, prices, counts, 'Negative Gaussian Curve Levels with TPO', 'negativeCurve', vah, val);

                    // Display data in widget
                    displayDataWidget(positiveLevels, negativeLevels);
                } else {
                    console.error('Error: Level data is missing.');
                }
            }
        })
        .catch(error => console.error('Error:', error));
    });

    function updateCandleChart(data, stockSymbol) {
        const dates = data.map(item => new Date(item.x));
        const opens = data.map(item => item.y[0]);
        const highs = data.map(item => item.y[1]);
        const lows = data.map(item => item.y[2]);
        const closes = data.map(item => item.y[3]);

        const minPrice = Math.min(...lows);
        const maxPrice = Math.max(...highs);
        const buffer = 10; // Adjust as needed
        const windowHeight = 500; // Adjust as needed

        const layout = {
            title: {
                text: `${stockSymbol} Stock Price`,
                font: { color: 'black' }
            },
            xaxis: {
                title: { text: 'Date', font: { color: 'black' } },
                rangeslider: { visible: false },
                tickfont: { color: 'black' }
            },
            yaxis: {
                title: { text: 'Price', standoff: 20, font: { color: 'black' } },
                tickfont: { color: 'black' },
                range: [minPrice - buffer, maxPrice + buffer]
            },
            height: windowHeight,
            margin: { t: 100, b: 100, l: 100, r: 50, pad: 10 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'rgba(0, 0, 0, 0.03)',
            showlegend: false,
            border: { color: 'black', width: 1 }
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

    function plotGaussianCurve(levels, prices, counts, title, divId, vah, val) {
        // Create traces for different opacity levels
        const traces = [
            {
                y: prices.filter(price => price >= vah),
                x: counts.filter((_, index) => prices[index] >= vah),
                orientation: 'h',
                type: 'bar',
                marker: { color: 'green', opacity: 1.0 },
                name: 'Above VAH'
            },
            {
                y: prices.filter(price => price > val && price < vah),
                x: counts.filter((_, index) => prices[index] > val && prices[index] < vah),
                orientation: 'h',
                type: 'bar',
                marker: { color: 'royalblue', opacity: 1.0 },
                name: 'Between VA'
            },
            {
                y: prices.filter(price => price <= val),
                x: counts.filter((_, index) => prices[index] <= val),
                orientation: 'h',
                type: 'bar',
                marker: { color: 'green', opacity: 1.0 },
                name: 'Below VAL'
            }
        ];
    
        // Add lines for levels
        for (const [levelName, levelValue] of Object.entries(levels)) {
            const color = levelName.includes('Strong') ? 'red' : levelName.includes('SELL') || levelName.includes('BUY') ? 'green' : levelName.includes('POC') ? 'purple' : levelName.includes('Low (Median)') || levelName.includes('High (Median)') ? '#FF6500' : 'orange';
            const dash = levelName.includes('POC') ? 'solid' : 'dash';
            traces.push({
                x: [0, Math.max(...counts)],
                y: [levelValue, levelValue],
                mode: 'lines',
                line: { color: color, dash: dash },
                name: `${levelName}: ${levelValue.toFixed(2)}`
            });
        }
    
        const layout = {
            title: title,
            xaxis: { title: 'TPO Count' },
            yaxis: { title: 'Price' },
            legend: { title: 'Levels' },
            template: 'plotly_white'
        };
    
        Plotly.newPlot(divId, traces, layout);
    }

    function displayDataWidget(positiveLevels, negativeLevels) {
        const widgetContainer = document.getElementById('dataWidget');
    
        const widgetContent = `
            
            <div class="level-container">
                <div class="positive-levels">
                    <h4>Positive Levels</h4>
                    <ul>
                        ${Object.entries(positiveLevels).map(([name, value]) => `
                            <li>
                                <span class="level-name">${name}:</span>
                                <span class="level-value">${value.toFixed(2)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                <div class="negative-levels">
                    <h4>Negative Levels</h4>
                    <ul>
                        ${Object.entries(negativeLevels).map(([name, value]) => `
                            <li>
                                <span class="level-name">${name}:</span>
                                <span class="level-value">${value.toFixed(2)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            </div>
        `;
    
        widgetContainer.innerHTML = widgetContent;
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
});