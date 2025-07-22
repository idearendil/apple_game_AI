import copy

import math
import random
import copy

class MCTSNode:
    def __init__(self, game, is_my_turn, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = None
        self.untried_moves = game.generateValidMoves()
        self.is_my_turn = is_my_turn

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        if self.is_my_turn:
            return max(
                self.children,
                key=lambda child: child.value / (child.visits + 1e-6) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )
        else:
            return max(
                self.children,
                key=lambda child: -child.value / (child.visits + 1e-6) - c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )

    def expand(self):
        move = self.untried_moves.pop()
        next_game = self.game.clone()
        next_game.updateMove(*move, True)
        child_node = MCTSNode(next_game, not self.is_my_turn, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        if self.is_my_turn:
            self.value = max(self.value, result) if self.value is not None else result
        else:
            self.value = min(self.value, result) if self.value is not None else result
        if self.parent:
            self.parent.backpropagate(self.value)  # 상대는 반대 결과

    def simulate(self):
        # sim_game = self.game.clone()
        # current_turn = True
        # for _ in range(0):  # 시뮬레이션 깊이 제한
        #     moves = sim_game.generateValidMoves()
        #     if not moves:
        #         break
        #     move = random.choice(moves)
        #     sim_game.updateMove(*move, current_turn)
        #     current_turn = not current_turn
        return (self.game.my_score - self.game.opp_score) / 100

class Game:
    def __init__(self, board, occupied_board, first, my_score=0, opp_score=0):
        self.board = board            # 게임 보드 (2차원 배열)
        self.occupied_board = occupied_board  # 점유된 칸 표시 (1: 내 차지, -1: 상대 차지, 0: 비어 있음)
        self.first = first            # 선공 여부
        self.passed = False           # 마지막 턴에 패스했는지 여부
        self.row_size = len(board)    # 보드의 행 크기
        self.col_size = len(board[0]) if board else 0  # 보드의 열 크기
        self.my_score = my_score        # 내 점수
        self.opp_score = opp_score        # 상대 점수
    
    def clone(self):
        return Game(copy.deepcopy(self.board), copy.deepcopy(self.occupied_board), self.first, self.my_score, self.opp_score)

    def mcts(self, _myTime, _oppTime, iter_limit=100):
        root = MCTSNode(self.clone(), True)

        for _ in range(iter_limit):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            result = node.simulate()

            # Backpropagation
            node.backpropagate(result)

        if not root.children:
            return (-1, -1, -1, -1)
        best = root.best_child(c_param=0)
        return best.move
    
    # 포함관계 제거 (작은 직사각형만 남기기)
    def is_contained(self, inner, outer):
        ir1, ic1, ir2, ic2 = inner
        or1, oc1, or2, oc2 = outer
        return or1 <= ir1 and oc1 <= ic1 and or2 >= ir2 and oc2 >= ic2

    # 현재 보드에서 가능한 모든 유효한 사각형 리스트 생성
    def generateValidMoves(self):
        # 누적합(dp) 계산
        dp = [[0] * self.col_size for _ in range(self.row_size)]
        dp[0][0] = self.board[0][0]
        for r in range(1, self.row_size):
            dp[r][0] = dp[r-1][0] + self.board[r][0]
        for c in range(1, self.col_size):
            dp[0][c] = dp[0][c-1] + self.board[0][c]
        for r in range(1, self.row_size):
            for c in range(1, self.col_size):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] - dp[r - 1][c - 1] + self.board[r][c]

                candidate_moves = []
        rows, cols = self.row_size, self.col_size
        for height in range(1, rows + 1):
            for r1 in range(rows - height + 1):
                r2 = r1 + height - 1
                c1 = c2 = 0
                while c2 < cols:
                    total = dp[r2][c2]
                    if r1 > 0:
                        total -= dp[r1 - 1][c2]
                    if c1 > 0:
                        total -= dp[r2][c1 - 1]
                    if r1 > 0 and c1 > 0:
                        total += dp[r1 - 1][c1 - 1]
                    if total >= 10:
                        if total == 10:
                            candidate_moves.append((r1, c1, r2, c2))
                        c1 += 1
                        if c2 < c1:
                            c2 = c1
                    else:
                        c2 += 1

        filtered_moves = []
        for i, move in enumerate(candidate_moves):
            containing = False
            for j, other in enumerate(candidate_moves):
                if i != j and self.is_contained(other, move):
                    containing = True
                    break
            if not containing:
                filtered_moves.append(move)

        return filtered_moves


    # 보드 상태 점수화 함수 (아직 비어 있음)
    def evaluate_move(self, move, _isMyMove=True):
        r1, c1, r2, c2 = move
        sums = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if r == 0 or c == 0 or r == self.row_size - 1 or c == self.col_size - 1:
                    if (r == 0 or r == self.row_size - 1) and (c == 0 or c == self.col_size - 1):
                        weight = 3
                    else:
                        weight = 2
                else:
                    weight = 1
                if self.occupied_board[r][c] == -1:
                    sums += weight
                if self.occupied_board[r][c] != 1:
                    sums += weight
        
        sums *= 1 if _isMyMove else -1
        return sums

    # 실제 AI가 수를 계산하는 함수
    def calculateMove(self, _myTime, _oppTime):
        return self.mcts(_myTime, _oppTime, iter_limit=9000)

    # 상대방의 수를 받아 보드에 반영
    def updateOpponentAction(self, action, _time):
        self.updateMove(*action, False)

    # 주어진 수를 보드에 반영 (칸을 0으로 지움)
    def updateMove(self, r1, c1, r2, c2, _isMyMove):
        if r1 == c1 == r2 == c2 == -1:
            self.passed = True
            return
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
                if r == 0 or c == 0 or r == self.row_size - 1 or c == self.col_size - 1:
                    if (r == 0 or r == self.row_size - 1) and (c == 0 or c == self.col_size - 1):
                        weight = 3
                    else:
                        weight = 2
                else:
                    weight = 1
                if _isMyMove:
                    if self.occupied_board[r][c] == -1:
                        self.opp_score -= weight
                    if self.occupied_board[r][c] != 1:
                        self.my_score += weight
                        self.occupied_board[r][c] = 1
                else:
                    if self.occupied_board[r][c] == 1:
                        self.my_score -= weight
                    if self.occupied_board[r][c] != -1:
                        self.opp_score += weight
                        self.occupied_board[r][c] = -1
        self.passed = False



# ================================
# main(): 입출력 처리 및 게임 진행
# ================================
def main():
    while True:
        line = input().split()

        if len(line) == 0:
            continue

        command, *param = line

        if command == "READY":
            # 선공 여부 확인
            turn = param[0]
            global first
            first = turn == "FIRST"
            print("OK", flush=True)
            continue

        if command == "INIT":
            # 보드 초기화
            board = [list(map(int, row)) for row in param]
            occupied_board = [[0] * len(board[0]) for _ in range(len(board))]
            global game
            game = Game(board, occupied_board, first)
            continue

        if command == "TIME":
            # 내 턴: 수 계산 및 실행
            myTime, oppTime = map(int, param)
            ret = game.calculateMove(myTime, oppTime)
            game.updateMove(*ret, True)
            print(*ret, flush=True)
            continue

        if command == "OPP":
            # 상대 턴 반영
            r1, c1, r2, c2, time = map(int, param)
            game.updateOpponentAction((r1, c1, r2, c2), time)
            continue

        if command == "FINISH":
            break

        assert False, f"Invalid command {command}"


if __name__ == "__main__":
    main()
