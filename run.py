import sys, time
import heapq
from dataclasses import dataclass
from typing import TypeAlias, Generator

cell_tuple: TypeAlias = tuple[int, int]

INF = 10**9
best_energy: dict[tuple[int, ...], int] = dict()
object_from_bitmap: dict[int, str] = {
    1: 'A',
    2: 'B',
    4: 'C',
    8: 'D',
    0: '',
}
object_to_bitmap: dict[str, int] = {
    'A': 1,
    'B': 2,
    'C': 4,
    'D': 8,
    '': 0,
}

# по позизции комнаты сопоставляется bitmap комнаты, заполненной объектами своего типа
FULL_ROOM_POS_BITMAP: dict[int, int] = {
    8: 34952, # 1000100010001000
    2: 4369, # 0001000100010001
    4: 8738, #0010001000100010
    6: 17476, #0100010001000100
}
ROOM_BY_BITMAP: dict[int, int] = {
    1: 2,
    2: 4,
    4: 6,
    8: 8,
}
ENERGY_BY_BITMAP: dict[int, int] = {
    1: 1,
    2: 10,
    4: 100,
    8: 1000
}

H_SHIFT = [(10 - y) << 2 for y in range(11)] # коридор 

object_names: list[str] = ['A', 'B', 'C', 'D']
room_position_numbers: list[int] = [2, 4, 6, 8]
depth_room = 0

@dataclass
class State:
    # representation: tuple[int, int, int, int, int] - побитовые маски: коридор и четыре комнаты
    # каждая ячейка будет состоять из 4 битов
    # если в ячейке:
    # - символ A, то 0001
    # - символ B, то 0010
    # - символ C, то 0100
    # - символ D, то 1000
    # - ничего, то 0000
    representation: tuple[int, int, int, int, int] # первые 11 элементов - это коридор, потом идут комнаты
    energy: int = 0
    
    def _get_obj_bitmap(self, obj_pos: cell_tuple) -> int:
        if obj_pos[0] == 0:
            offset, index = (10 - obj_pos[1]) << 2, 0
        else:
            offset, index = (depth_room  - obj_pos[0]) << 2, obj_pos[1] >> 1
        return (self.representation[index] >> (offset)) & 0b1111
    
    def _objects_in_room_target_type(self, room_pos: int) -> bool:
        return (FULL_ROOM_POS_BITMAP[room_pos] & self.representation[room_pos >> 1]) == self.representation[room_pos >> 1]
    
    
    def end_state(self) -> bool:
        return self.representation[0] == 0 and \
                all(self._objects_in_room_target_type(room_pos) for room_pos in room_position_numbers)
    
    def generate_next_states(self) -> Generator['State', None, None]:
        for obj_cell in self._get_object_cells_in_hallway():
            obj_bitmap = self._get_obj_bitmap(obj_cell)
            target_room = ROOM_BY_BITMAP[obj_bitmap]
            top_obj_in_room_cell = self._get_top_empty_in_room_cell(target_room)
            if self._can_enter_room(obj_cell):
                yield self._move_object(obj_cell, top_obj_in_room_cell)

        for room_pos in room_position_numbers:
            obj_cell = self._get_top_obj_in_room_cell(room_pos)
            if obj_cell is None:
                continue
            obj_bitmap = self._get_obj_bitmap(obj_cell)
            if self._objects_in_room_target_type(room_pos):
                    continue
            
            available_hallway_cells = ((0, i) for i in range(11) if i not in room_position_numbers)
            for cell in available_hallway_cells:
                if self._can_move_in_hallway(obj_cell, cell):
                    yield self._move_object(obj_cell, cell)

            if self._can_enter_room(obj_cell):
                target_cell = self._get_top_empty_in_room_cell(ROOM_BY_BITMAP[obj_bitmap])
                yield self._move_object(obj_cell, target_cell)

    # Эвристика для алгоритма A*
    def calculate_heuristic(self) -> int:
        total_cost = 0
        hallway_bits = self.representation[0]
        for y in range(11):
            nib = (hallway_bits >> H_SHIFT[y]) & 0b1111
            if nib != 0:
                target_x = ROOM_BY_BITMAP[nib]
                total_cost += (abs(y - target_x) + 1) * ENERGY_BY_BITMAP[nib]

        for room_x in room_position_numbers:
            room_bits = self.representation[room_x >> 1]
            for k in range(depth_room - 1, -1, -1):
                nib = (room_bits >> (4 * k)) & 0b1111
                if nib != 0:
                    cur_depth = depth_room - k
                    target_x = ROOM_BY_BITMAP[nib]
                    if target_x != room_x:
                        total_cost += (cur_depth + abs(room_x - target_x) + 1) * ENERGY_BY_BITMAP[nib]
        return total_cost

    #TODO
    def _move_object(self, cur_pos: cell_tuple, to_pos: cell_tuple) -> 'State':
        new_representation = list(self.representation)

        def _set_bitmap(pos: cell_tuple, bitmap: int):
            if pos[0] == 0:
                arr_index = 0
                bit_pos = H_SHIFT[pos[1]]
            else:
                arr_index = pos[1] >> 1
                bit_pos = (depth_room - pos[0]) << 2
            clear_mask = ~(0b1111 << bit_pos)
            new_representation[arr_index] = (new_representation[arr_index] & clear_mask) | ((bitmap & 0b1111) << bit_pos)

        bitmap = self._get_obj_bitmap(cur_pos)
        _set_bitmap(to_pos, bitmap)
        _set_bitmap(cur_pos, 0)

        dist = abs(to_pos[1] - cur_pos[1]) + to_pos[0] + cur_pos[0]
        wasted = ENERGY_BY_BITMAP[bitmap] * dist

        return State(representation=tuple(new_representation),  # type: ignore
                     energy=self.energy + wasted)
    
    
    def _get_objects_in_room(self, room_number: int) -> list[int]:
        bits = self.representation[room_number >> 1]
        return [
            nib
            for depth in range(depth_room - 1, -1, -1)
            if (nib := (bits >> (depth << 2)) & 0b1111)
        ]

    
    def _get_object_cells_in_hallway(self) -> Generator[cell_tuple, None, None]:
        hall = self.representation[0]
        return ((0, y) for y in range(11) if ((hall >> ((10 - y) << 2)) & 0b1111))

    def _get_top_empty_in_room_cell(self, room_number: int) -> cell_tuple:
        top_obj_in_room_cell = self._get_top_obj_in_room_cell(room_number)
        if top_obj_in_room_cell is None:
            return depth_room, room_number
        return top_obj_in_room_cell[0] - 1, room_number
    
    def _get_top_obj_in_room_cell(self, room_number: int) -> cell_tuple | None:
        objects_in_room = self._get_objects_in_room(room_number)
        if len(objects_in_room) == 0:
            return None
        return depth_room - len(objects_in_room) + 1, room_number

    def _can_enter_room(self, obj_cell: cell_tuple) -> bool:
        obj_bitmap = self._get_obj_bitmap(obj_cell)
        room_pos = ROOM_BY_BITMAP[obj_bitmap]
        return self._objects_in_room_target_type(room_pos) and self._can_move_in_hallway(obj_cell, (0, room_pos))
    
    def _can_move_in_hallway(self, src_cell: cell_tuple, target_cell: cell_tuple) -> bool:
        y0, y1 = src_cell[1], target_cell[1]
        step = 1 if y1 > y0 else -1
        return all((self.representation[0] >> H_SHIFT[i]) & 0b1111 == 0 for i in range(y0 + step, y1 + step, step))

    def get_representation(self) -> tuple[int, ...]:
        return self.representation
    
    def __lt__(self, other: 'State'):
        return self.energy < other.energy


def solve(lines: list[str]) -> int:
    """
    Решение задачи о сортировке в лабиринте

    Args:
        lines: список строк, представляющих лабиринт

    Returns:
        минимальная энергия для достижения целевой конфигурации
    """
    # TODO: Реализация алгоритма
    heap: list[tuple[int, State]] = []
    result = INF
    repr_state = get_start_representation_state(lines)
    heapq.heappush(heap, (0, State(repr_state)))
    while len(heap) > 0:
        _, cur_state = heapq.heappop(heap)
        current_positions = cur_state.get_representation()
        if best_energy.get(current_positions, INF) < cur_state.energy:
            continue
        best_energy[cur_state.get_representation()] = cur_state.energy
        if cur_state.end_state():
            result = min(result, cur_state.energy)
            continue
        for next_state in cur_state.generate_next_states():
            next_positions = next_state.get_representation()
            if next_state.energy < best_energy.get(next_positions, INF):
                best_energy[next_positions] = next_state.energy
                next_f_cost = next_state.energy + next_state.calculate_heuristic()
                heapq.heappush(heap, (next_f_cost, next_state))
    return int(result)



def get_start_representation_state(lines) -> tuple[int, int, int, int, int]:
    global depth_room
    depth_room = len(lines) - 3
    hallway = 0
    for i in range(1, 12):
        temp = lines[1][i] if lines[1][i] != '.' else ''
        hallway |= ((object_to_bitmap[temp]) << ((11 - i) << 2))
    
    rooms = [0, 0, 0, 0]
    for i in (3, 5, 7, 9):
        for j in range(2, 2 + depth_room):
            temp = lines[j][i] if lines[j][i] != '.' else ''
            rooms[i // 2 - 1] |= ((object_to_bitmap[temp]) << ((depth_room + 1 - j) << 2))
    return hallway, rooms[0], rooms[1], rooms[2], rooms[3]


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()