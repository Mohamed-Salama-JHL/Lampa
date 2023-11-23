from abc import ABC, abstractmethod


class analysis_abstract(ABC):
    @abstractmethod
    def __init__(self,dataset,dataset_controls) -> None:
        pass
    @abstractmethod    
    def get_page(self):
        pass
    @abstractmethod    
    def get_dataset_new(self):
        pass
    @abstractmethod    
    def get_update_button(self):
        pass

    @abstractmethod    
    def create_sidebar(self):
        pass
    @abstractmethod    
    def create_main_area(self):
        pass