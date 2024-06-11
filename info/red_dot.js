document.addEventListener('DOMContentLoaded', (event) => {
    const gameArea = document.getElementById('gameArea');
    const target = document.getElementById('target');
    const scoreDisplay = document.getElementById('score');
    const averageTimeDisplay = document.getElementById('averageTime');
    let score = 0;
    let clickTimes = [];
    let lastClickTime = null;
    let gameStarted = false;

    function moveTarget() {
        const gameAreaRect = gameArea.getBoundingClientRect();
        const maxX = gameAreaRect.width - target.clientWidth;
        const maxY = gameAreaRect.height - target.clientHeight;

        const randomX = Math.floor(Math.random() * maxX);
        const randomY = Math.floor(Math.random() * maxY);

        target.style.left = `${randomX}px`;
        target.style.top = `${randomY}px`;
    }

    function calculateAverageTime(times) {
        const sum = times.reduce((a, b) => a + b, 0);
        return sum / times.length;
    }

    target.addEventListener('click', () => {
        const currentTime = new Date().getTime();
        if (gameStarted) {
            const timeDifference = currentTime - lastClickTime;
            clickTimes.push(timeDifference);
        } else {
            gameStarted = true;
        }
        lastClickTime = currentTime;

        score++;
        scoreDisplay.textContent = score;

        if (score === 21) {
            const averageTime = calculateAverageTime(clickTimes);
            alert(`Avg time: ${averageTime.toFixed(2)} ms`);
            score = 0;
            scoreDisplay.textContent = score;
            clickTimes = [];
            gameStarted = false;
            averageTimeDisplay.textContent = '';
        }

        moveTarget();
    });

    moveTarget();
});