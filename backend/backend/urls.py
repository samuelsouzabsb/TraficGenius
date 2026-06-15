"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os
import mimetypes
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.static import serve

# Corrige problema de MIME type em sistemas Windows
mimetypes.add_type("application/javascript", ".js", True)
mimetypes.add_type("text/css", ".css", True)

# Caminho absoluto para a pasta frontend (peer da pasta backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    
    # Redireciona a raiz / para index.html
    path('', serve, {'document_root': FRONTEND_DIR, 'path': 'index.html'}),
    
    # Serve qualquer outro arquivo da pasta frontend (como styles.css, app.js, mapa_completo.html, etc.)
    re_path(r'^(?P<path>.*)$', serve, {'document_root': FRONTEND_DIR}),
]
