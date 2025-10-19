import sys, time
import heapq
from dataclasses import dataclass
from typing import TypeAlias, Generator, Literal

cell: TypeAlias = tuple[int, int]
frozenset_cells: TypeAlias = frozenset[cell]
object_name_type: TypeAlias = Literal['A'] | Literal['B'] | Literal['C'] | Literal['D']

@dataclass
class Obj:
    name: object_name_type 
    curPos: cell
    energy: int
    moved: bool = False

    def move(self, to: cell) -> tuple['Obj', int]:
        wastedEnergy = self.energy * (abs(to[1] - self.curPos[1]) + to[0] + self.curPos[0])
        return Obj(
            name=self.name,
            curPos=to,
            energy=self.energy,
            moved=True
        ), wastedEnergy

INF = 10**9
list_objects: TypeAlias = list[Obj]
best_energy: dict[tuple[frozenset_cells, ...], int] = dict()

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
room_numbers: list[int] = [2, 4, 6, 8]

@dataclass
class State:
    objects: list_objects
    energy: int = 0

    def end_state(self) -> bool:
        return all(obj.curPos[1] == object_room[obj.name] for obj in self.objects)
    
    def next_states(self) -> Generator['State', None, None]:
        objects_in_hallway = self._get_objects_in_hallway()
        for obj in objects_in_hallway:
            top_cell_room = self._get_top_room_cell(object_room[obj.name])
            if self._can_enter_room(obj) and self._can_move_in_hallway(obj, (0, top_cell_room[1])):
                yield self._move_object(obj, top_cell_room)
        for room_number in room_numbers:
            obj = self._get_top_room_obj(room_number)
            if obj is None or obj.moved:
                continue
            if obj.curPos[1] == object_room[obj.name] and \
                all(x.name == obj.name for x in self._get_objects_in_room(room_number)):
                    continue
            hallway_cells_without_room = [(0, i) for i in range(11) if i not in room_numbers]
            for c in hallway_cells_without_room:
                if self._can_move_in_hallway(obj, c):
                    yield self._move_object(obj, c)
            if self._can_enter_room(obj) and self._can_move_in_hallway(obj, (0, object_room[obj.name])):
                yield self._move_object(obj, self._get_top_room_cell(object_room[obj.name]))

    # Эвристика для алгоритма A*
    def calculate_heuristic(self) -> int:
        total_cost = 0
        for obj in self._get_objects_in_hallway():
            room_pos = object_room[obj.name]
            total_cost += abs(obj.curPos[1] - room_pos) + 1
        for room_number in room_numbers:
            objects_in_room = self._get_objects_in_room(room_number)
            for i, obj in enumerate(objects_in_room):
                if object_room[obj.name] != room_number:
                    room_pos = object_room[obj.name]
                    current_room_pos = room_number
                    max_depth = len(self.objects) // 4
                    depth = max_depth - len(objects_in_room) + i
                    total_cost += object_energy[obj.name] * (abs(current_room_pos - room_pos) + depth + 2)
        return total_cost

    def _move_object(self, obj: Obj, to: cell) -> 'State':
        new_objects = list(self.objects)
        new_obj, wasted_energy = obj.move(to)
        for i in range(len(new_objects)):
            if new_objects[i] == obj:
                new_objects[i] = new_obj
        return State(
            objects=new_objects,
            energy=self.energy + wasted_energy
        )
    
    def _get_objects_in_room(self, room_number: int) -> list[Obj]:
        return sorted([obj for obj in self.objects if obj.curPos[1] == room_number], key=lambda x: -x.curPos[0])
    
    def _get_objects_in_hallway(self) -> list[Obj]:
        return sorted([obj for obj in self.objects if obj.curPos[0] == 0], key=lambda x: x.curPos[1])
    
    def _get_top_room_obj(self, room_number: int) -> Obj | None:
        objects_in_room = self._get_objects_in_room(room_number)
        if len(objects_in_room) == 0:
            return None
        return objects_in_room[-1]

    def _get_top_room_cell(self, room_number: int) -> cell:
        depth = len(self.objects) // 4
        objects_in_room = self._get_objects_in_room(room_number)
        return depth - len(objects_in_room), room_number 

    def _can_enter_room(self, obj: Obj) -> bool:
        room_number = object_room[obj.name]
        return all(x.name == obj.name for x in self._get_objects_in_room(room_number)) and \
              self._can_move_in_hallway(obj, (0, room_number))
    
    def _can_move_in_hallway(self, obj: Obj, hallway_cell: cell) -> bool:
        objects_in_hallway = self._get_objects_in_hallway()
        min_coord = min(hallway_cell[1], obj.curPos[1])
        max_coord = max(hallway_cell[1], obj.curPos[1])
        return len([x for x in objects_in_hallway if x != obj and min_coord <= x.curPos[1] <= max_coord]) == 0


    def get_object_positions(self) -> tuple[frozenset_cells, ...]:
        return tuple(
            [frozenset([obj.curPos for obj in self.objects if obj.name == obj_name]) for obj_name in object_names])
    
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
    heapq.heappush(heap, (0, State(get_objs(lines))))
    while len(heap) > 0:
        _, cur_state = heapq.heappop(heap)
        current_positions = cur_state.get_object_positions()
        if best_energy.get(current_positions, INF) < cur_state.energy:
            continue
        best_energy[cur_state.get_object_positions()] = cur_state.energy
        if cur_state.end_state():
            result = min(result, cur_state.energy)
            continue
        for next_state in cur_state.next_states():
            next_positions = next_state.get_object_positions()
            if next_state.energy < best_energy.get(next_positions, INF):
                best_energy[next_positions] = next_state.energy
                next_f_cost = next_state.energy + next_state.calculate_heuristic()
                heapq.heappush(heap, (next_f_cost, next_state))
    return int(result)



def get_objs(lines) -> list[Obj]:
    obj = list()
    for i in range(1, len(lines) - 1):
        for j in range(len(lines[i])):
            if lines[i][j] in object_energy:
                obj.append(Obj(
                    name=lines[i][j],
                    curPos=(i - 1, j - 1),
                    energy=object_energy[lines[i][j]]
                ))
    return obj


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()