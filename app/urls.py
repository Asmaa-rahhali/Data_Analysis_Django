from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'), 
    path('probabilites/', views.probabilites_view, name='probabilites'),
]