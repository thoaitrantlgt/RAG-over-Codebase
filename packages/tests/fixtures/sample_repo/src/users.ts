export function getUser(id: string) {
  return { id };
}

class UserService {
  findUser(id: string) {
    return getUser(id);
  }
}

export class ExportedService {
  run() {
    return true;
  }
}
