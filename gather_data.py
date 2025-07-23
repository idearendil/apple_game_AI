import copy

class Game:

    def __init__(self, board, occupied_board, first, my_score=0, opp_score=0, passed=False):
        self.board = board            # 게임 보드 (2차원 배열)
        self.occupied_board = occupied_board  # 점유된 칸 표시 (1: 내 차지, -1: 상대 차지, 0: 비어 있음)
        self.first = first            # 선공 여부
        self.passed = passed           # 마지막 턴에 패스했는지 여부
        self.ended = False           # 게임 종료 여부
        self.row_size = len(board)    # 보드의 행 크기
        self.col_size = len(board[0]) if board else 0  # 보드의 열 크기
        self.my_score = my_score        # 내 점수
        self.opp_score = opp_score        # 상대 점수

    # 사각형 (r1, c1) ~ (r2, c2)이 유효한지 검사 (합이 10이고, 네 변을 모두 포함)
    def isValid(self, r1, c1, r2, c2):
        sums = 0
        r1fit = c1fit = r2fit = c2fit = False

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] != 0:
                    sums += self.board[r][c]
                    if r == r1:
                        r1fit = True
                    if r == r2:
                        r2fit = True
                    if c == c1:
                        c1fit = True
                    if c == c2:
                        c2fit = True
        return sums == 10 and r1fit and r2fit and c1fit and c2fit
    
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

        # 합이 정확히 10인 직사각형 찾기
        candidate_moves = []
        rows, cols = self.row_size, self.col_size
        for height in range(1, rows + 1):
            break_flag = True
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
                        break_flag = False
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
        
        filtered_moves.append((-1, -1, -1, -1))  # 패스 가능한 경우 추가

        return filtered_moves


    # 보드 상태 점수화 함수 (아직 비어 있음)
    def evaluate(self):
        return self.my_score - self.opp_score

    # 알파-베타 가지치기 적용 minimax 알고리즘
    def minimax(self, now_game, depth, isMax, alpha=float("-inf"), beta=float("inf")):
        # 깊이 제한 또는 게임 종료 시점 도달
        if depth == 0 or now_game.ended:
            return now_game.evaluate(), (-1, -1, -1, -1)

        moves = now_game.generateValidMoves()

        if not moves:
            return now_game.evaluate(), (-1, -1, -1, -1)

        best_move = (-1, -1, -1, -1)

        if isMax:
            maxEval = float("-inf")
            for move in moves:
                new_board = copy.deepcopy(now_game.board)
                new_occupied_board = copy.deepcopy(now_game.occupied_board)
                temp_game = Game(new_board, new_occupied_board, self.first, now_game.my_score, now_game.opp_score, now_game.passed)
                _ = temp_game.updateMove(*move, True)
                eval, _ = self.minimax(temp_game, depth - 1, False, alpha, beta)
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # 가지치기
            return maxEval, best_move
        else:
            minEval = float("inf")
            for move in moves:
                new_board = copy.deepcopy(now_game.board)
                new_occupied_board = copy.deepcopy(now_game.occupied_board)
                temp_game = Game(new_board, new_occupied_board, self.first, now_game.my_score, now_game.opp_score, now_game.passed)
                _ = temp_game.updateMove(*move, False)
                eval, _ = self.minimax(temp_game, depth - 1, True, alpha, beta)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # 가지치기
            return minEval, best_move

    # 실제 AI가 수를 계산하는 함수
    def calculateMove(self, _myTime, _oppTime, turn_cnt):
        moves = self.generateValidMoves()
        if len(moves) > 12:
            depth = 3
        elif len(moves) > 7:
            depth = 4
        else:
            depth = 7
        temp_game = Game(self.board, self.occupied_board, self.first, self.my_score, self.opp_score, self.passed)
        _, best_move = self.minimax(temp_game, depth, True)
        return best_move, len(moves)

    # 주어진 수를 보드에 반영 (칸을 0으로 지움)
    def updateMove(self, r1, c1, r2, c2, _isMyMove):
        eaten_cnt = 0
        if r1 == c1 == r2 == c2 == -1:
            if self.passed:
                self.ended = True
            self.passed = True
            return eaten_cnt
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
                if r == 0 or c == 0 or r == self.row_size - 1 or c == self.col_size - 1:
                    if (r == 0 or r == self.row_size - 1) and (c == 0 or c == self.col_size - 1):
                        weight = 2
                    else:
                        weight = 1.5
                else:
                    weight = 1
                if _isMyMove:
                    if self.occupied_board[r][c] == -1:
                        eaten_cnt += 1
                        self.opp_score -= weight * 2
                    if self.occupied_board[r][c] != 1:
                        self.my_score += weight
                        self.occupied_board[r][c] = 1
                else:
                    if self.occupied_board[r][c] == 1:
                        eaten_cnt += 1
                        self.my_score -= weight * 2
                    if self.occupied_board[r][c] != -1:
                        self.opp_score += weight
                        self.occupied_board[r][c] = -1
        self.passed = False
        return eaten_cnt



# ================================
# main(): 입출력 처리 및 게임 진행
# ================================
import random
import pickle

R = 10
C = 17
repeat_num = 10000

def flip_ones(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == -1:
                matrix[i][j] = 1
            elif matrix[i][j] == 1:
                matrix[i][j] = -1
    return matrix

def count_from_matrix(matrix, value):
    count_num = 0
    for row in matrix:
        for val in row:
            if val == value:
                count_num += 1
    return count_num

def edge_count_from_matrix(matrix, value):
    height = len(matrix)
    width = len(matrix[0])

    count_num = 0
    for row_idx, row in enumerate(matrix):
        for val_idx, val in enumerate(row):
            if row_idx == 0 or row_idx == height - 1 or val_idx == 0 or val_idx == width - 1:
                if val == value:
                    count_num += 1
    return count_num

def corner_count_from_matrix(matrix, value):
    height = len(matrix)
    width = len(matrix[0])

    count_num = 0
    for row_idx, row in enumerate(matrix):
        for val_idx, val in enumerate(row):
            if (row_idx == 0 or row_idx == height - 1) and (val_idx == 0 or val_idx == width - 1):
                if val == value:
                    count_num += 1
    return count_num

def who_win(matrix):
    count_neg1 = count_from_matrix(matrix, -1)
    count_pos1 = count_from_matrix(matrix, 1)
    return 0 if count_neg1 < count_pos1 else 1

def main():
    turn_cnt = 0
    for repeat_idx in range(repeat_num):
        print(f"Repeat {repeat_idx + 1} / {repeat_num}")
        # 보드 초기화
        board = [[random.randint(0, 9) for _ in range(C)] for _ in range(R)]
        occupied_board = [[0] * len(board[0]) for _ in range(len(board))]
        game = Game(board, occupied_board, True)

        data1 = []
        data2 = []

        eaten_cnt1 = eaten_cnt2 = 0

        while not game.ended:
            turn_cnt += 1

            ret, _ = game.calculateMove(0, 0, turn_cnt)
            eaten_cnt1 += game.updateMove(*ret, True)
            _, moves_num1 = game.calculateMove(0, 0, turn_cnt)
            area_num1 = count_from_matrix(game.occupied_board, 1)
            edge_num1 = edge_count_from_matrix(game.occupied_board, 1)
            corner_num1 = corner_count_from_matrix(game.occupied_board, 1)
            area_num2 = count_from_matrix(game.occupied_board, -1)
            edge_num2 = edge_count_from_matrix(game.occupied_board, -1)
            corner_num2 = corner_count_from_matrix(game.occupied_board, -1)

            data1.append([
                turn_cnt,
                0,
                moves_num1,
                area_num1,
                edge_num1,
                corner_num1,
                area_num2,
                edge_num2,
                corner_num2,
                eaten_cnt1,
                eaten_cnt2
            ])

            if game.ended:
                break

            game.occupied_board = flip_ones(game.occupied_board)
            
            ret, _ = game.calculateMove(0, 0, turn_cnt)
            eaten_cnt2 += game.updateMove(*ret, True)
            _, moves_num2 = game.calculateMove(0, 0, turn_cnt)
            area_num1 = count_from_matrix(game.occupied_board, -1)
            edge_num1 = edge_count_from_matrix(game.occupied_board, -1)
            corner_num1 = corner_count_from_matrix(game.occupied_board, -1)
            area_num2 = count_from_matrix(game.occupied_board, 1)
            edge_num2 = edge_count_from_matrix(game.occupied_board, 1)
            corner_num2 = corner_count_from_matrix(game.occupied_board, 1)

            data2.append([
                turn_cnt,
                1,
                moves_num2,
                area_num2,
                edge_num2,
                corner_num2,
                area_num1,
                edge_num1,
                corner_num1,
                eaten_cnt2,
                eaten_cnt1
            ])
            
            game.occupied_board = flip_ones(game.occupied_board)
        
        won_player = who_win(game.occupied_board)
        if won_player == 0:
            for sample in data1:
                sample.append(1)
            for sample in data2:
                sample.append(0)
        else:
            for sample in data1:
                sample.append(0)
            for sample in data2:
                sample.append(1)

        with open(f'data/lst{repeat_idx}.pkl', 'wb') as f:
            pickle.dump(data1 + data2, f)


if __name__ == "__main__":
    main()
