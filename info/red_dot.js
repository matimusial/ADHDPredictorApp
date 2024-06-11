document.addEventListener('DOMContentLoaded', (event) => {
    const gameArea = document.getElementById('gameArea');
    const target = document.getElementById('target');
    const scoreDisplay = document.getElementById('score');
    let score = 0;

    function moveTarget() {
        const gameAreaRect = gameArea.getBoundingClientRect();
        const maxX = gameAreaRect.width - target.clientWidth;
        const maxY = gameAreaRect.height - target.clientHeight;

        const randomX = Math.floor(Math.random() * maxX);
        const randomY = Math.floor(Math.random() * maxY);

        target.style.left = `${randomX}px`;
        target.style.top = `${randomY}px`;
    }

    target.addEventListener('click', () => {
        score++;
        scoreDisplay.textContent = score;
        moveTarget();
    });

    moveTarget();
});