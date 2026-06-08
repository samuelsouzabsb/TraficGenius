from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AcidenteViewSet, DashboardStatsView, PredictionView

# Utiliza o roteador padrão do DRF para o ViewSet de acidentes
router = DefaultRouter()
router.register(r'acidentes', AcidenteViewSet, basename='acidente')

urlpatterns = [
    # Inclui as rotas do roteador (/api/acidentes/ e /api/acidentes/<id>/)
    path('', include(router.urls)),
    
    # Rota para as estatisticas do Dashboard (/api/dashboard-stats/)
    path('dashboard-stats/', DashboardStatsView.as_view(), name='dashboard-stats'),
    
    # Rota para predição de severidade em tempo real (/api/predict/)
    path('predict/', PredictionView.as_view(), name='predict'),
]
