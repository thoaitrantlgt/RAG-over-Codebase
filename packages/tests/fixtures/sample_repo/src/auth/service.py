import jwt


def check_permission(user, resource):
    return user.can_access(resource)


class AuthService:
    def verify_token(self, token):
        payload = jwt.decode(token)
        return payload["sub"]

    def refresh_token(self, user_id):
        return f"refresh:{user_id}"
