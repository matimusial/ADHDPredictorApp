const canvas = document.getElementById('pongCanvas');
const ctx = canvas.getContext('2d');

canvas.width = 800;
canvas.height = 400;

const paddleWidth = 10;
const paddleHeight = 100;
const ballRadius = 10;

const playerPaddle = {
    x: 10,
    y: canvas.height / 2 - paddleHeight / 2,
    width: paddleWidth,
    height: paddleHeight,
    dy: 0,
    speed: 6
};

const computerPaddle = {
    x: canvas.width - 10 - paddleWidth,
    y: canvas.height / 2 - paddleHeight / 2,
    width: paddleWidth,
    height: paddleHeight,
    dy: 2, // Zmniejszenie prędkości komputera
    reactionDelay: 0 // Opóźnienie reakcji komputera
};

const ball = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    radius: ballRadius,
    dx: 4,
    dy: 4
};

let playerScore = 0;
let computerScore = 0;
const winningScore = 5;
let isGameRunning = false;

const ballImg = new Image();
ballImg.src = 'pilka.png';

const startButton = document.getElementById('startButton');
startButton.addEventListener('click', () => {
    resetGame();
    isGameRunning = true;
    loop();
});

function drawRect(x, y, width, height) {
    ctx.fillStyle = '#fff';
    ctx.fillRect(x, y, width, height);
}

function drawImage(img, x, y, width, height) {
    ctx.drawImage(img, x, y, width, height);
}

function drawText(text, x, y) {
    ctx.fillStyle = '#fff';
    ctx.font = '20px Arial';
    ctx.fillText(text, x, y);
}

function update() {
    if (!isGameRunning) return;

    // Move player paddle
    playerPaddle.y += playerPaddle.dy;

    // Prevent player paddle from going out of bounds
    if (playerPaddle.y < 0) {
        playerPaddle.y = 0;
    }
    if (playerPaddle.y + playerPaddle.height > canvas.height) {
        playerPaddle.y = canvas.height - playerPaddle.height;
    }

    // Move computer paddle with reaction delay
    if (computerPaddle.reactionDelay <= 0) {
        if (ball.y < computerPaddle.y + computerPaddle.height / 2) {
            computerPaddle.y -= computerPaddle.dy;
        } else {
            computerPaddle.y += computerPaddle.dy;
        }
        // Losowe opóźnienie reakcji komputera
        computerPaddle.reactionDelay = Math.random() * 10 + 10;
    } else {
        computerPaddle.reactionDelay--;
    }

    // Prevent computer paddle from going out of bounds
    if (computerPaddle.y < 0) {
        computerPaddle.y = 0;
    }
    if (computerPaddle.y + computerPaddle.height > canvas.height) {
        computerPaddle.y = canvas.height - computerPaddle.height;
    }

    // Move ball
    ball.x += ball.dx;
    ball.y += ball.dy;

    // Ball collision with top and bottom walls
    if (ball.y - ball.radius < 0 || ball.y + ball.radius > canvas.height) {
        ball.dy *= -1;
    }

    // Ball collision with paddles
    if (ball.x - ball.radius < playerPaddle.x + playerPaddle.width &&
        ball.y > playerPaddle.y &&
        ball.y < playerPaddle.y + playerPaddle.height) {
        ball.dx *= -1;
        ball.x = playerPaddle.x + playerPaddle.width + ball.radius; // Zapobiega zapadnięciu się piłki w paddle
    }

    if (ball.x + ball.radius > computerPaddle.x &&
        ball.y > computerPaddle.y &&
        ball.y < computerPaddle.y + computerPaddle.height) {
        ball.dx *= -1;
        ball.x = computerPaddle.x - ball.radius; // Zapobiega zapadnięciu się piłki w paddle
    }

    // Reset ball and update score if it goes out of bounds
    if (ball.x - ball.radius < 0) {
        computerScore++;
        resetBall();
    }
    if (ball.x + ball.radius > canvas.width) {
        playerScore++;
        resetBall();
    }

    // Check for winning score
    if (playerScore >= winningScore || computerScore >= winningScore) {
        isGameRunning = false;
        setTimeout(() => {
            resetGame();
            isGameRunning = true;
        }, 2000);
    }
}

function resetBall() {
    ball.x = canvas.width / 2;
    ball.y = canvas.height / 2;
    ball.dx = (Math.random() > 0.5 ? 1 : -1) * 4;
    ball.dy = (Math.random() > 0.5 ? 1 : -1) * 4;
}

function resetGame() {
    playerScore = 0;
    computerScore = 0;
    resetBall();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawRect(playerPaddle.x, playerPaddle.y, playerPaddle.width, playerPaddle.height);
    drawRect(computerPaddle.x, computerPaddle.y, computerPaddle.width, computerPaddle.height);
    drawImage(ballImg, ball.x - ball.radius, ball.y - ball.radius, ball.radius * 2, ball.radius * 2);
    drawText(`Brain: ${playerScore}`, 20, 20);
    drawText(`Computer: ${computerScore}`, canvas.width - 150, 20);
}

function loop() {
    update();
    draw();
    if (isGameRunning) {
        requestAnimationFrame(loop);
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') {
        playerPaddle.dy = -playerPaddle.speed;
    } else if (e.key === 'ArrowDown') {
        playerPaddle.dy = playerPaddle.speed;
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        playerPaddle.dy = 0;
    }
});
