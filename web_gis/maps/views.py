from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def route_map(request):
    return render(request, 'route-map.html')