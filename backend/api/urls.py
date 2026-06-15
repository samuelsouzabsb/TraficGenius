from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    AcidenteViewSet, DashboardStatsView, PredictionView, 
    DashboardStatsView3Classes, PredictionView3Classes,
    DFH3ListView, PredictH3View, PredictGridView
)

# Utiliza o roteador padrão do DRF para o ViewSet de acidentes
router = DefaultRouter()
router.register(r'acidentes', AcidenteViewSet, basename='acidente')

urlpatterns = [
    # Inclui as rotas do roteador (/api/acidentes/ e /api/acidentes/<id>/)
    path('', include(router.urls)),
    
    # Rota para as estatisticas do Dashboard (/api/dashboard-stats/)
    path('dashboard-stats/', DashboardStatsView.as_view(), name='dashboard-stats'),
    
    # Rota para as estatisticas do Dashboard - 3 Classes (/api/dashboard-stats-3classes/)
    path('dashboard-stats-3classes/', DashboardStatsView3Classes.as_view(), name='dashboard-stats-3classes'),
    
    # Rota para predição de severidade em tempo real (/api/predict/)
    path('predict/', PredictionView.as_view(), name='predict'),
    
    # Rota para predição de severidade em tempo real - 3 Classes (/api/predict-3classes/)
    path('predict-3classes/', PredictionView3Classes.as_view(), name='predict-3classes'),
    
    # Novas rotas para Distrito Federal (DF)
    path('df-h3/', DFH3ListView.as_view(), name='df-h3'),
    path('predict-h3/', PredictH3View.as_view(), name='predict-h3'),
    path('predict-grid/', PredictGridView.as_view(), name='predict-grid'),
]

