document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const randomBtn = document.getElementById('randomBtn');
    const mt5Btn = document.getElementById('mt5Btn');
    const textRankBtn = document.getElementById('textRankBtn');
    const evaluateBtn = document.getElementById('evaluateBtn');
    
    const articleElem = document.getElementById('article');
    const referenceElem = document.getElementById('reference');
    const mt5SummaryElem = document.getElementById('mt5Summary');
    const textRankSummaryElem = document.getElementById('textRankSummary');
    const evaluationResults = document.getElementById('evaluationResults');
    const evaluationChart = document.getElementById('evaluationChart');
    
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    
    // 状态变量
    let hasMt5Summary = false;
    let hasTextRankSummary = false;
    
    // 显示加载中状态
    function showLoading(message) {
        loadingText.textContent = message || '处理中...';
        loadingOverlay.style.display = 'flex';
    }
    
    // 隐藏加载中状态
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
    
    // 随机选择文章
    randomBtn.addEventListener('click', function() {
        showLoading('正在随机选择文章...');
        
        fetch('/random_article')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    articleElem.textContent = data.article;
                    referenceElem.textContent = data.reference;
                    mt5SummaryElem.textContent = '';
                    textRankSummaryElem.textContent = '';
                    evaluationResults.textContent = '';
                    evaluationChart.style.display = 'none';
                    
                    // 启用摘要生成按钮
                    mt5Btn.disabled = false;
                    textRankBtn.disabled = false;
                    
                    // 重置状态
                    hasMt5Summary = false;
                    hasTextRankSummary = false;
                    evaluateBtn.disabled = true;
                } else {
                    alert('获取文章失败: ' + data.message);
                }
            })
            .catch(error => {
                console.error('错误:', error);
                alert('获取文章时发生错误');
            })
            .finally(() => {
                hideLoading();
            });
    });
    
    // 生成mT5摘要
    mt5Btn.addEventListener('click', function() {
        showLoading('正在生成mT5摘要，这可能需要一些时间...');
        
        fetch('/generate_mt5', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    mt5SummaryElem.textContent = data.mt5_summary;
                    hasMt5Summary = true;
                    
                    // 检查是否可以启用评估按钮
                    if (hasMt5Summary && hasTextRankSummary) {
                        evaluateBtn.disabled = false;
                    }
                } else {
                    alert('生成mT5摘要失败: ' + data.message);
                }
            })
            .catch(error => {
                console.error('错误:', error);
                alert('生成mT5摘要时发生错误');
            })
            .finally(() => {
                hideLoading();
            });
    });
    
    // 生成TextRank摘要
    textRankBtn.addEventListener('click', function() {
        showLoading('正在生成TextRank摘要...');
        
        fetch('/generate_textrank', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    textRankSummaryElem.textContent = data.textrank_summary;
                    hasTextRankSummary = true;
                    
                    // 检查是否可以启用评估按钮
                    if (hasMt5Summary && hasTextRankSummary) {
                        evaluateBtn.disabled = false;
                    }
                } else {
                    alert('生成TextRank摘要失败: ' + data.message);
                }
            })
            .catch(error => {
                console.error('错误:', error);
                alert('生成TextRank摘要时发生错误');
            })
            .finally(() => {
                hideLoading();
            });
    });
    
    // 评估结果
    evaluateBtn.addEventListener('click', function() {
        showLoading('正在评估摘要质量...');
        
        fetch('/evaluate', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 显示评估文本结果
                    let resultHTML = '<h3>评分结果</h3>';
                    resultHTML += '<table border="1" style="width:100%; border-collapse: collapse; margin-top:10px;">';
                    resultHTML += '<tr><th>方法</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th></tr>';
                    
                    for (const [method, scores] of Object.entries(data.results)) {
                        resultHTML += `<tr>
                            <td>${method}</td>
                            <td>${scores.rouge1.toFixed(4)}</td>
                            <td>${scores.rouge2.toFixed(4)}</td>
                            <td>${scores.rougeL.toFixed(4)}</td>
                        </tr>`;
                    }
                    resultHTML += '</table>';
                    
                    evaluationResults.innerHTML = resultHTML;
                    
                    // 显示评估图表
                    evaluationChart.src = 'data:image/png;base64,' + data.image;
                    evaluationChart.style.display = 'block';
                } else {
                    alert('评估失败: ' + data.message);
                }
            })
            .catch(error => {
                console.error('错误:', error);
                alert('评估时发生错误');
            })
            .finally(() => {
                hideLoading();
            });
    });
});
