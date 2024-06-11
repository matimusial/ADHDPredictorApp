document.addEventListener('DOMContentLoaded', (event) => {
    const X_CLASS = 'x';
    const CIRCLE_CLASS = 'circle';
    const WINNING_COMBINATIONS = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ];

    const cellElements = document.querySelectorAll('[data-cell]');
    const board = document.getElementById('ticTacToeBoard');
    const winningMessageElement = document.getElementById('winningMessage');
    const restartButton = document.getElementById('restartButton');
    const winnerElement = document.getElementById('winner');
    let circleTurn;

    startGame();

    restartButton.addEventListener('click', startGame);

    function startGame() {
        circleTurn = true; // Ensure the player always starts
        cellElements.forEach(cell => {
            cell.classList.remove(X_CLASS);
            cell.classList.remove(CIRCLE_CLASS);
            cell.innerHTML = '';
            cell.addEventListener('click', handleClick, { once: true });
        });
        setBoardHoverClass();
        winningMessageElement.classList.add('hidden');
    }

    function handleClick(e) {
        const cell = e.target;
        if (cell.classList.contains(X_CLASS) || cell.classList.contains(CIRCLE_CLASS)) {
            return; // Ignore clicks on occupied cells
        }
        const currentClass = circleTurn ? CIRCLE_CLASS : X_CLASS;
        placeMark(cell, currentClass);
        if (checkWin(currentClass)) {
            endGame(false);
        } else if (isDraw()) {
            endGame(true);
        } else {
            swapTurns();
            setBoardHoverClass();
            if (!circleTurn) {
                makeBestMove();
            }
        }
    }

    function endGame(draw) {
        if (draw) {
            winnerElement.innerText = 'Draw!';
        } else {
            winnerElement.innerText = `${circleTurn ? "Brain" : "Computer"} wins!`;
        }
        winningMessageElement.classList.remove('hidden');
    }

    function isDraw() {
        return [...cellElements].every(cell => {
            return cell.classList.contains(X_CLASS) || cell.classList.contains(CIRCLE_CLASS);
        });
    }

    function placeMark(cell, currentClass) {
        if (currentClass === CIRCLE_CLASS) {
            const img = document.createElement('img');
            img.src = 'pilka.png';
            cell.appendChild(img);
        } else {
            cell.innerText = 'X';
        }
        cell.classList.add(currentClass);
    }

    function swapTurns() {
        circleTurn = !circleTurn;
    }

    function setBoardHoverClass() {
        board.classList.remove(X_CLASS);
        board.classList.remove(CIRCLE_CLASS);
        if (circleTurn) {
            board.classList.add(CIRCLE_CLASS);
        } else {
            board.classList.add(X_CLASS);
        }
    }

    function checkWin(currentClass) {
        return WINNING_COMBINATIONS.some(combination => {
            return combination.every(index => {
                return cellElements[index].classList.contains(currentClass);
            });
        });
    }

    function makeBestMove() {
        if (Math.random() < 0.2) { // 20% szansa na losowy ruch
            const randomMove = getRandomMove();
            const cell = cellElements[randomMove];
            placeMark(cell, X_CLASS);
        } else {
            const bestMove = minimax([...cellElements], X_CLASS);
            const cell = cellElements[bestMove.index];
            placeMark(cell, X_CLASS);
        }
        if (checkWin(X_CLASS)) {
            endGame(false);
        } else if (isDraw()) {
            endGame(true);
        } else {
            swapTurns();
            setBoardHoverClass();
        }
    }

    function getRandomMove() {
        const availableCells = [];
        cellElements.forEach((cell, index) => {
            if (!cell.classList.contains(X_CLASS) && !cell.classList.contains(CIRCLE_CLASS)) {
                availableCells.push(index);
            }
        });
        const randomIndex = Math.floor(Math.random() * availableCells.length);
        return availableCells[randomIndex];
    }

    function minimax(newBoard, player) {
        const availSpots = newBoard.filter(cell => !cell.classList.contains(X_CLASS) && !cell.classList.contains(CIRCLE_CLASS));

        if (checkWinAI(newBoard, CIRCLE_CLASS)) {
            return { score: -10 };
        } else if (checkWinAI(newBoard, X_CLASS)) {
            return { score: 10 };
        } else if (availSpots.length === 0) {
            return { score: 0 };
        }

        const moves = [];
        for (let i = 0; i < availSpots.length; i++) {
            const move = {};
            move.index = newBoard.indexOf(availSpots[i]);
            newBoard[move.index].classList.add(player);

            if (player === X_CLASS) {
                const result = minimax(newBoard, CIRCLE_CLASS);
                move.score = result.score;
            } else {
                const result = minimax(newBoard, X_CLASS);
                move.score = result.score;
            }

            newBoard[move.index].classList.remove(player);
            moves.push(move);
        }

        let bestMove;
        if (player === X_CLASS) {
            let bestScore = -Infinity;
            for (let i = 0; i < moves.length; i++) {
                if (moves[i].score > bestScore) {
                    bestScore = moves[i].score;
                    bestMove = i;
                }
            }
        } else {
            let bestScore = Infinity;
            for (let i = 0; i < moves.length; i++) {
                if (moves[i].score < bestScore) {
                    bestScore = moves[i].score;
                    bestMove = i;
                }
            }
        }

        return moves[bestMove];
    }

    function checkWinAI(board, player) {
        return WINNING_COMBINATIONS.some(combination => {
            return combination.every(index => {
                return board[index].classList.contains(player);
            });
        });
    }
});