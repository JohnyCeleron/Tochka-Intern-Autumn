import sys, time
import heapq
from dataclasses import dataclass
from typing import TypeAlias, Generator

cell_tuple: TypeAlias = tuple[int, int]

INF = 10**9
best_energy: dict[tuple[int, ...], int] = dict()
object_room: dict[str, int] = {
    'A': 2,
    'B': 4,
    'C': 6,
    'D': 8
}
object_energy: dict[str, int] = {
    'A': 1,
    'B': 10,
    'C': 100,
    'D': 1000
}
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
bitmap_full_room_pos: dict[int, int] = {
    8: 34952, # 1000100010001000
    2: 4369, # 0001000100010001
    4: 8738, #0010001000100010
    6: 17476, #0100010001000100
}
hallway_bitmap = 16492674416640 # 11110000000000000000000000000000000000000000

object_names: list[str] = ['A', 'B', 'C', 'D']
room_position_numbers: list[int] = [2, 4, 6, 8]
depth_room = 0

@dataclass
class State:
    #TODO: representation: tuple[int, int, int, int, int] - побитовые маски: коридор и четыре комнаты
    # каждая ячейка будет состоять из 4 битов
    # если в ячейке:
    # - символ A, то 0001
    # - символ B, то 0010
    # - символ C, то 0100
    # - символ D, то 1000
    # - ничего, то 0000
    representation: tuple[int, int, int, int, int] # первые 11 элементов - это коридор, потом идут комнаты
    energy: int = 0

    def _get_obj_name(self, obj_pos: cell_tuple) -> str:
        return object_from_bitmap[self._get_obj_bitmap(obj_pos)]
    
    def _get_obj_bitmap(self, obj_pos: cell_tuple) -> int:
        
        if obj_pos[0] == 0:
            result = (self.representation[0] >> ((10 - obj_pos[1]) * 4)) & 0b1111
            return result
        room_index = (obj_pos[1] >> 1)
        return (self.representation[room_index] >> ((depth_room  - obj_pos[0]) * 4)) & 0b1111
    
    def _objects_in_room_target_type(self, room_pos: int) -> bool:
        return (bitmap_full_room_pos[room_pos] & self.representation[room_pos >> 1]) == self.representation[room_pos >> 1]

    def _waste_energy(self, cur_pos: cell_tuple, to: cell_tuple) -> int:
        obj_name = self._get_obj_name(cur_pos)
        return object_energy[obj_name] * (abs(to[1] - cur_pos[1]) + to[0] + cur_pos[0])
    
    
    def end_state(self) -> bool:
        return self.representation[0] == 0 and \
                all(self._objects_in_room_target_type(room_pos) for room_pos in room_position_numbers)
    
    #TODO
    def generate_next_states(self) -> Generator['State', None, None]:
        for obj_cell in self._get_object_cells_in_hallway():
            obj_name = self._get_obj_name(obj_cell)
            target_room = object_room[obj_name]
            top_obj_in_room_cell = self._get_top_empty_in_room_cell(target_room)
            if self._can_enter_room(obj_cell):
                yield self._move_object(obj_cell, top_obj_in_room_cell)

        for room_pos in room_position_numbers:
            obj_cell = self._get_top_obj_in_room_cell(room_pos)
            if obj_cell is None:
                continue
            obj_name = self._get_obj_name(obj_cell)
            if self._objects_in_room_target_type(room_pos):
                    continue
            
            available_hallway_cells = ((0, i) for i in range(11) if i not in room_position_numbers)
            for cell in available_hallway_cells:
                if self._can_move_in_hallway(obj_cell, cell):
                    yield self._move_object(obj_cell, cell)

            if self._can_enter_room(obj_cell):
                target_cell = self._get_top_empty_in_room_cell(object_room[obj_name])
                yield self._move_object(obj_cell, target_cell)

    # Эвристика для алгоритма A*
    #TODO
    def calculate_heuristic(self) -> int:
        total_cost = 0
        for obj_cell in self._get_object_cells_in_hallway():
            obj_name = self._get_obj_name(obj_cell)
            target_room = object_room[obj_name]
            distance = abs(obj_cell[1] - target_room) + 1
            total_cost += distance * object_energy[obj_name]


        for room_x in room_position_numbers:
            objects_in_room = self._get_objects_in_room(room_x)
            objects_in_room_cells = [(depth_room - len(objects_in_room) + depth + 1, room_x) for depth, _ in enumerate(objects_in_room)]
            for obj_cell in objects_in_room_cells:
                obj_name = self._get_obj_name(obj_cell)
                target_room_x = object_room[obj_name]
                if target_room_x != room_x:
                    current_depth = obj_cell[0]
                    min_distance = current_depth + abs(room_x - target_room_x) + 1
                    total_cost += min_distance * object_energy[obj_name]
        return total_cost

    #TODO
    def _move_object(self, cur_pos: cell_tuple, to_pos: cell_tuple) -> 'State':
        new_representation = list(self.representation)

        def _set_bitmap(pos: cell_tuple, bitmap: int):
            if pos[0] == 0:
                bit_pos = (10 - pos[1]) * 4
                clear_mask = ~(0b1111 << bit_pos)
                new_representation[0] &= clear_mask
                new_representation[0] |= (bitmap & 0b1111) << bit_pos
                #time.sleep(2)
            else:
                arr_index = pos[1] // 2
                bit_pos = (depth_room - pos[0]) * 4
                clear_mask = ~(0b1111 << bit_pos)
                new_representation[arr_index] &= clear_mask
                new_representation[arr_index] |= (bitmap & 0b1111) << bit_pos

        bitmap = self._get_obj_bitmap(cur_pos)
        _set_bitmap(to_pos, bitmap)
        _set_bitmap(cur_pos, 0)
        wasted_energy = self._waste_energy(cur_pos, to_pos)
        return State(
            representation=tuple(new_representation),  # type: ignore
            energy=self.energy + wasted_energy
        )
    
    
    def _get_objects_in_room(self, room_number: int) -> list[int]:
        return [self._get_obj_bitmap((i + 1, room_number)) for i in range(depth_room) \
                if self._get_obj_bitmap((i + 1, room_number)) != 0]

    
    def _get_object_cells_in_hallway(self) -> Generator[cell_tuple, None, None]:
        return ((0, i) for i in range(11) if self._get_obj_bitmap((0, i)) != 0)

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
        obj_name = self._get_obj_name(obj_cell)
        room_pos = object_room[obj_name]
        return self._objects_in_room_target_type(room_pos) and self._can_move_in_hallway(obj_cell, (0, room_pos))
    
    def _can_move_in_hallway(self, src_cell: cell_tuple, target_cell: cell_tuple) -> bool:
        y0, y1 = src_cell[1], target_cell[1]
        if y0 == y1:
            # Нужна только проверка двери (клетка (0, y0) должна быть свободна) — её ниже охватит общий цикл.
            pass
        step = 1 if y1 > y0 else -1
        # Проверяем ВСЕ клетки коридора по пути, исключая стартовую, включая целевую.
        for y in range(y0 + step, y1 + step, step):
            if self._get_obj_bitmap((0, y)) != 0:
                return False
        return True

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