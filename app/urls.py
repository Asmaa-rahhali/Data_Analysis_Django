from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Page d'accueil
    path('probabilites/', views.probabilites_view, name='probabilites'),
]