from abc import ABCMeta, abstractmethod


class DecisionMaker(metaclass=ABCMeta):

    @abstractmethod
    def get_action(self, measurement, time: float = 0.0, **kwargs):
        pass

    @abstractmethod
    def update_energies(self, measurement, costs: dict, time: float = 0.0, **kwargs) -> None:
        pass

