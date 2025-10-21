import sys, time
import heapq
from dataclasses import dataclass
from typing import TypeAlias, Generator, Literal

cell_tuple: TypeAlias = tuple[int, int]
cell_frozenset: TypeAlias = frozenset[cell_tuple]
object_name_type: TypeAlias = Literal['A'] | Literal['B'] | Literal['C'] | Literal['D']

@dataclass
class Obj:
    name: object_name_type 
    curPos: cell_tuple
    energy: int
    moved: bool = False

    def move(self, to: cell_tuple) -> tuple['Obj', int]:
        wastedEnergy = self.energy * (abs(to[1] - self.curPos[1]) + to[0] + self.curPos[0])
        return Obj(
            name=self.name,
            curPos=to,
            energy=self.energy,
            moved=True
        ), wastedEnergy

INF = 10**9
list_objects: TypeAlias = list[Obj]
best_energy: dict[tuple[str, ...], int] = dict()

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

object_names: list[object_name_type] = ['A', 'B', 'C', 'D']
room_position_numbers: list[int] = [2, 4, 6, 8]

@dataclass
class State:
    representation: tuple[str, ...] # первые 11 элементов - это коридор, потом идут комнаты
    energy: int = 0

    def _get_depth_room(self) -> int:
        return (len(self.representation) - 11) // 4
    
    def _get_obj_name(self, obj_pos: cell_tuple) -> str:
        return self.representation[self._get_representation_index(obj_pos)]
    
    def _waste_energy(self, cur_pos: cell_tuple, to: cell_tuple) -> int:
        obj_name = self._get_obj_name(cur_pos)
        return object_energy[obj_name] * (abs(to[1] - cur_pos[1]) + to[0] + cur_pos[0])
    
    def _get_representation_index(self, pos: cell_tuple) -> int:
        depth = self._get_depth_room()
        if pos[0] == 0:
            return pos[1]
        return 11 + (pos[1] // 2 - 1) * depth + (pos[0] - 1)
    
    def end_state(self) -> bool:
        depth = self._get_depth_room()
        return all(self.representation[11 + depth * room_number + cur_depth] == obj_name \
                    for cur_depth in range(depth) \
                    for room_number, obj_name in enumerate(object_names))
    
    def generate_next_states(self) -> Generator['State', None, None]:
        for obj_cell in self._get_object_cells_in_hallway():
            obj_name = self._get_obj_name(obj_cell)
            target_room = object_room[obj_name]
            top_obj_in_room_cell = self._get_top_empty_in_room_cell(target_room)
            if self._can_enter_room(obj_cell) and self._can_move_in_hallway(obj_cell, (0, top_obj_in_room_cell[1])):
                yield self._move_object(obj_cell, top_obj_in_room_cell)

        for room_pos in room_position_numbers:
            obj_cell = self._get_top_obj_in_room_cell(room_pos)
            
            if obj_cell is None:
                continue
            
            obj_name = self._get_obj_name(obj_cell)
            if obj_cell[1] == object_room[obj_name] and \
                all(x == "" or x == obj_name for x in self._get_object_names_in_room(room_pos)):
                    continue
            
            available_hallway_cells = [(0, i) for i in range(11) if i not in room_position_numbers]
            for cell in available_hallway_cells:
                if self._can_move_in_hallway(obj_cell, cell):
                    yield self._move_object(obj_cell, cell)

            if self._can_enter_room(obj_cell) and self._can_move_in_hallway(obj_cell, (0, object_room[obj_name])):
                target_cell = self._get_top_empty_in_room_cell(object_room[obj_name])
                yield self._move_object(obj_cell, target_cell)

    # Эвристика для алгоритма A*
    def calculate_heuristic(self) -> int:
        total_cost = 0
        for obj_cell in self._get_object_cells_in_hallway():
            obj_name = self._get_obj_name(obj_cell)
            target_room = object_room[obj_name]
            distance = abs(obj_cell[1] - target_room) + 1
            total_cost += distance * object_energy[obj_name]


        for room_x in room_position_numbers:
            objects_in_room = [(depth + 1, room_x) for depth, obj_name \
                               in enumerate(self._get_object_names_in_room(room_x))
                               if obj_name != '']
            for obj_cell in objects_in_room:
                obj_name = self._get_obj_name(obj_cell)
                target_room_x = object_room[obj_name]
                if target_room_x != room_x:
                    current_depth = obj_cell[0]
                    min_distance = current_depth + abs(room_x - target_room_x) + 1
                    total_cost += min_distance * object_energy[obj_name]
        return total_cost

    def _move_object(self, cur_pos: cell_tuple, to_pos: cell_tuple) -> 'State':
        new_representation = list(self.representation)
        from_index, to_index = self._get_representation_index(cur_pos), self._get_representation_index(to_pos)
        new_representation[from_index], new_representation[to_index] = \
            new_representation[to_index], new_representation[from_index]
        wasted_energy = self._waste_energy(cur_pos, to_pos)
        return State(
            representation=tuple(new_representation),
            energy=self.energy + wasted_energy
        )
    
    def _get_object_names_in_room(self, room_number: int) -> list[str]:
        list_representation = list(self.representation)
        depth = self._get_depth_room()
        start_index = 11 + depth * (room_number // 2 - 1)
        return list_representation[start_index: start_index + depth]

    
    def _get_object_cells_in_hallway(self) -> Generator[cell_tuple, None, None]:
        return ((0, i) for i in range(11) if self.representation[i] != '')

    def _get_top_empty_in_room_cell(self, room_number: int) -> cell_tuple:
        max_depth = self._get_depth_room()
        top_obj_in_room_cell = self._get_top_obj_in_room_cell(room_number)
        if top_obj_in_room_cell is None:
            return max_depth, room_number
        return top_obj_in_room_cell[0] - 1, room_number
    
    def _get_top_obj_in_room_cell(self, room_number: int) -> cell_tuple | None:
        objects_in_room = self._get_object_names_in_room(room_number)
        max_depth = self._get_depth_room()
        top_depth = 0
        for depth in range(max_depth):
            if objects_in_room[depth] != '':
                break
            top_depth += 1
        if top_depth == max_depth:
            return None
        return top_depth + 1, room_number

    def _can_enter_room(self, obj_cell: cell_tuple) -> bool:
        obj_name = self._get_obj_name(obj_cell)
        room_number = object_room[obj_name]
        return all(x == '' or x == obj_name for x in self._get_object_names_in_room(room_number)) and \
              self._can_move_in_hallway(obj_cell, (0, room_number))
    
    def _can_move_in_hallway(self, obj_cell: cell_tuple, hallway_cell: cell_tuple) -> bool:
        min_coord = min(hallway_cell[1], obj_cell[1])
        max_coord = max(hallway_cell[1], obj_cell[1])
        obj_name = self._get_obj_name(obj_cell)
        return all(self.representation[i] == '' for i in range(11) \
                   if self.representation[i] != obj_name and min_coord <= i <= max_coord)


    def get_representation(self) -> tuple[str, ...]:
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
    heapq.heappush(heap, (0, State(get_start_representation_state(lines))))
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



def get_start_representation_state(lines) -> tuple[str, ...]:
    hallway = []
    for i in range(1, 12):
        temp = lines[1][i] if lines[1][i] != '.' else ''
        hallway.append(temp)
    depth = len(lines) - 3
    rooms = [[], [], [], []]
    for i in (3, 5, 7, 9):
        for j in range(2, 2 + depth):
            temp = lines[j][i] if lines[j][i] != '.' else ''
            rooms[i // 2 - 1].append(temp)
    return tuple(hallway + rooms[0] + rooms[1] + rooms[2] + rooms[3])


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()