export const exampleQueries = [
  {
    category: 'connectivity',
    label: 'Connectivity',
    icon: '🔗',
    queries: [
      'Why can\'t 10.0.0.1 connect to 10.0.1.1?',
      'Trace path from 10.0.0.1 to 10.0.1.1',
      'Is traffic blocked from 10.0.0.1 to 10.0.1.1 on TCP 443?',
      'Check routing between 192.168.1.1 and 192.168.2.1',
    ],
  },
  {
    category: 'device',
    label: 'Device',
    icon: '🖥️',
    queries: [
      'arista1 is dropping packets',
      'Check OSPF neighbors on arista2',
      'Show interface errors on arista1',
      'Why is arista3 unreachable?',
    ],
  },
  {
    category: 'incidents',
    label: 'ServiceNow',
    icon: '🎫',
    queries: [
      'Troubleshoot INC0010035',
      'What open incidents are related to arista1?',
      'Are there any recent changes on the core routers?',
    ],
  },
]
