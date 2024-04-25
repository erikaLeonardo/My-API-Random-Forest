from django.urls import path
from .views import index  # Cambiar homeView a index

urlpatterns = [
    path('', index, name='home')  # Cambiar homeView a index
]
