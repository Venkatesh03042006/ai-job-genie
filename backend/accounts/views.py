from django.contrib.auth import authenticate, login, logout
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.serializers import LoginSerializer, RegisterSerializer, UserSerializer


class RegisterView(APIView):
    """POST /api/auth/register/ - create a user and log them in immediately
    (session cookie set on the response), so the frontend doesn't need a
    separate login call right after registering.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        login(request, user)
        return Response(UserSerializer(user).data, status=201)


class LoginView(APIView):
    """POST /api/auth/login/ - authenticate and start a session."""
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = authenticate(
            request,
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            return Response({"detail": "Invalid username or password."}, status=400)

        login(request, user)
        return Response(UserSerializer(user).data, status=200)


class LogoutView(APIView):
    """POST /api/auth/logout/ - end the current session."""
    permission_classes = [AllowAny]

    def post(self, request):
        logout(request)
        return Response(status=204)


@method_decorator(ensure_csrf_cookie, name="get")
class MeView(APIView):
    """GET /api/auth/me/ - returns the logged-in user, or null for anonymous
    visitors. Also seeds the csrftoken cookie so the frontend has a CSRF
    token available before it needs to POST anywhere.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        if not request.user.is_authenticated:
            return Response({"user": None})
        return Response({"user": UserSerializer(request.user).data})
